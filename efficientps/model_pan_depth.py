import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import MeanSquaredError, JaccardIndex
from .fpn import TwoWayFpn
from .backbone import generate_backbone_EfficientPS, output_feature_size
from .semantic_head import SemanticHead
from .depth_head import DepthHead
from .refine_head import RefineHead
from .instance_head import InstanceHead
from .panoptic_segmentation_module import scale_resize_pad_masks, check_bbox_size
from .panoptic_segmentation_module import panoptic_segmentation_module
from .panoptic_metrics import generate_pred_panoptic
from .panoptic_predictions import panoptic_predictions
from panopticapi.evaluation import pq_compute
import os.path

class Pan_Depth(pl.LightningModule):
    """
    EfficientPS model see http://panoptic.cs.uni-freiburg.de/
    Here pytorch lightningis used https://pytorch-lightning.readthedocs.io/en/latest/
    """

    def __init__(self, cfg, categories, lr=None):
        """
        Args:
        - cfg (Config) : Config object from detectron2
        """
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        if lr is None:
            self.learning_rate = cfg.SOLVER.BASE_LR_PAN_DEPTH
        else:
            self.learning_rate = lr

        self.backbone = generate_backbone_EfficientPS(cfg)
        self.fpn = TwoWayFpn(
            output_feature_size[cfg.MODEL_CUSTOM.BACKBONE.EFFICIENTNET_ID])

        self.instance_head = InstanceHead(cfg)
        self.semantic_head = SemanticHead(cfg.NUM_CLASS)
        self.depth_head = DepthHead(
            cfg.VKITTI_DATASET.DEPTH.K, 
            num_classes=cfg.NUM_CLASS,
            n_points=cfg.VKITTI_DATASET.DEPTH.MAX_DEPTH_POINTS)
        
        self.refine_head = RefineHead(cfg.NUM_CLASS)

        self.valid_acc_depth = MeanSquaredError(squared=False)
        self.valid_acc_sem = JaccardIndex(cfg.NUM_CLASS)
        # self.valid_acc_bbx = MeanAveragePrecision(class_metrics=True)
        self.obj_categories=categories
    

    def on_load_checkpoint(self, checkpoint):
        checkpoint.pop("optimizer_states")
        
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        predictions, _ = self.shared_step(x)
        return predictions

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        _, loss = self.shared_step(batch)
        
        # Add losses to logs
        [self.log("{}_step".format(k), v, batch_size=self.cfg.BATCH_SIZE, on_step=True, on_epoch=False, sync_dist=False) for k,v in loss.items()]
        [self.log(k, v, batch_size=self.cfg.BATCH_SIZE, on_step=False, on_epoch=True, sync_dist=True) for k,v in loss.items()]
        self.log('train_loss', sum(loss.values()), batch_size=self.cfg.BATCH_SIZE, on_step=True, on_epoch=False, sync_dist=False)
        self.log('train_loss_epoch', sum(loss.values()), batch_size=self.cfg.BATCH_SIZE, on_step=False, on_epoch=True, sync_dist=True)
        return {'loss': sum(loss.values())}

    def shared_step(self, inputs):
        loss = dict()
        predictions = dict()
        img = inputs['image']
        semantic_gt = inputs["semantic"]
        mask = inputs['mask']
        coors = inputs['virtual_lidar']
        k_nn_indices = inputs['k_nn_indices']
        sparse_depth = inputs['sparse_depth']
        sparse_depth_gt = inputs['sparse_depth_gt']
        # depth_full = inputs['depth_full']

        # Feature extraction
        features = self.backbone.extract_endpoints(img)
        pyramid_features = self.fpn(features)
        # Semantic Predictions
        output_size = inputs["image"][0].shape[-2:]
        semantic_logits, semantic_loss = self.semantic_head(pyramid_features, output_size, inputs)
        
        #Instance Predictions
        pred_instance, instance_losses = self.instance_head(pyramid_features, inputs)

        # Depth Predictions
        depth, depth_loss = self.depth_head(img,
            sparse_depth, 
            mask, 
            coors, 
            k_nn_indices,
            semantic_logits=semantic_logits, 
            sparse_depth_gt=sparse_depth_gt)

        # Refine Head
        refined_logits, refine_loss = self.refine_head(
            semantic_logits, 
            depth,
            # depth_full, 
            output_size, 
            semantic_gt=semantic_gt)
        
        # Output set up
        loss.update(depth_loss)
        loss.update(semantic_loss)
        loss.update(refine_loss)
        loss.update(instance_losses)
        loss.update({"loss_sum": depth_loss["depth_loss"] + refine_loss["refine_loss"]})

        predictions.update({'depth': depth})
        predictions.update({'semantic': refined_logits})
        # predictions.update({'semantic': semantic_logits})
        predictions.update({'instance': pred_instance})
        return predictions, loss

    def validation_step(self, batch, batch_idx):
        
        # depth -----------
        sparse_depth_gt = batch['sparse_depth_gt']

        mask_pos = torch.tensor((1), dtype=torch.float64, device=self.device)
        mask_neg = torch.tensor((0), dtype=torch.float64, device=self.device)
        mask_gt = torch.where(sparse_depth_gt > 0, mask_pos, mask_neg)
        mask_gt = mask_gt.squeeze_(1)
        
        predictions, _ = self.shared_step(batch)

        pred_depth = torch.squeeze(predictions["depth"], 1)*mask_gt
        # --------------------------
        #semantic
        pred_semantic = F.softmax(predictions["semantic"], dim=1)

        # Metrics Semantic
        self.valid_acc_sem(pred_semantic, batch["semantic"])
        self.log('IoU', self.valid_acc_sem, on_step=False, on_epoch=True, sync_dist=True)

        # Merics Depth
        self.valid_acc_depth(pred_depth, sparse_depth_gt.squeeze_(1)*mask_gt)
        self.log('RMSE', self.valid_acc_depth, on_step=False, on_epoch=True, sync_dist=True)

        # # Instance

        # target = [dict(
        #         boxes=instance.get("gt_boxes").tensor,
        #         labels=instance.get("gt_classes"),
        #         masks=instance.get("gt_masks").tensor
        #     ) for instance in batch["instance"]]
        
        # if predictions["instance"] != None:       
            
        #     instances = [check_bbox_size(instance) for instance in predictions["instance"]]
        #     preds = [dict(
        #         boxes=instance.get("pred_boxes").tensor,
        #         labels=instance.get("pred_classes"),
        #         masks=scale_resize_pad_masks(instance).to(torch.bool),
        #         scores=instance.get("scores")
        #     ) for instance in instances]
            
        # else:
        #     preds = [dict(
        #         boxes=torch.zeros((0, 4), dtype=torch.float).to(self.device),
        #         labels=torch.zeros((0), dtype=torch.long).to(self.device),
        #         masks=torch.zeros((0), dtype=torch.bool).to(self.device),
        #         scores=torch.zeros((0), dtype=torch.float).to(self.device)
        #     ) for _ in batch["instance"]]
        
        # # Metric
        # # self.valid_acc_bbx(preds, target)
        # # self.log_dict(self.valid_acc_bbx, on_step=False, on_epoch=True, sync_dist=True)
        # self.valid_acc_bbx.update(preds, target)

        #Panoptic
        panoptic_result = panoptic_segmentation_module(self.cfg,
            predictions,
            self.device)

        return {
            'panoptic': panoptic_result,
            'image_id': batch['image_id']
        }

    def validation_epoch_end(self, outputs):
        
         # BBoxes
        # mAPs = {"val_" + k: v for k, v in self.valid_acc_bbx.compute().items()}
        # # print(mAPs)
        # mAPs_per_class = mAPs.pop("val_map_per_class")
        # mARs_per_class = mAPs.pop("val_mar_100_per_class")
        # self.log_dict(mAPs, sync_dist=True)
        # self.log_dict(
        #     {
        #         f"val_map_{label}": value
        #         for label, value in zip(self.obj_categories.values(), mAPs_per_class)
        #     },
        #     sync_dist=True,
        # )
        # self.log_dict(
        #     {
        #         f"val_mar_100_{label}": value
        #         for label, value in zip(self.obj_categories.values(), mARs_per_class)
        #     },
        #     sync_dist=True,
        # )
        # self.valid_acc_bbx.reset()

        # print(outputs)
        gathered_results = self.all_gather(outputs)
        # print("gathered_results", gathered_results)
        if self.local_rank == 0:

            for gr in gathered_results:
                #Flatten results
                gr["image_id"] = [im_id for sublist in gr["image_id"] for im_id in sublist]
                gr["panoptic"] = [pan for sublist in gr["panoptic"] for pan in sublist]
                

            #1. convert coco gt to panoptic gt
            #2. Detections to coco format
            #3. Detections coco format to panoptic format
            #4. Evaluate
            # Create and save all predictions files
            generate_pred_panoptic(self.cfg, gathered_results)

            # Compute PQ metric with panpticapi
            pq_res = pq_compute(
                gt_json_file= os.path.join(
                    self.cfg.VKITTI_DATASET.DATASET_PATH.ROOT,
                    self.cfg.VKITTI_DATASET.DATASET_PATH.VALID_JSON),
                pred_json_file= os.path.join(
                    self.cfg.VKITTI_DATASET.DATASET_PATH.ROOT,
                    self.cfg.VKITTI_DATASET.DATASET_PATH.PRED_JSON),
                gt_folder= os.path.join(self.cfg.VKITTI_DATASET.DATASET_PATH.ROOT),
                pred_folder=os.path.join(
                    self.cfg.VKITTI_DATASET.DATASET_PATH.ROOT,
                    self.cfg.VKITTI_DATASET.DATASET_PATH.VALID_PRED_DIR)
            )
            
            self.log("PQ", 100 * pq_res["All"]["pq"] * self.cfg.NUM_GPUS, sync_dist=True)
            self.log("SQ", 100 * pq_res["All"]["sq"], rank_zero_only=True)
            self.log("RQ", 100 * pq_res["All"]["rq"], rank_zero_only=True)
            self.log("PQ_th", 100 * pq_res["Things"]["pq"], rank_zero_only=True)
            self.log("SQ_th", 100 * pq_res["Things"]["sq"], rank_zero_only=True)
            self.log("RQ_th", 100 * pq_res["Things"]["rq"], rank_zero_only=True)
            self.log("PQ_st", 100 * pq_res["Stuff"]["pq"], rank_zero_only=True)
            self.log("SQ_st", 100 * pq_res["Stuff"]["sq"], rank_zero_only=True)
            self.log("RQ_st", 100 * pq_res["Stuff"]["rq"], rank_zero_only=True)
        
        else:
            self.log("PQ", 0, sync_dist=True)


    # def predict_step(self, batch, batch_idx, dataloader_idx=0):

    #     predictions = dict()
    #     # Feature extraction
    #     features = self.backbone.extract_endpoints(batch['image'])
    #     pyramid_features = self.fpn(features)
    #     # Heads Predictions
    #     output_size = batch["image"][0].shape[-2:]
    #     semantic_logits, _ = self.semantic_head(pyramid_features, output_size)

    #     predictions.update({'semantic': semantic_logits})
    #     preds = F.softmax(predictions["semantic"], dim=1)
    #     preds = F.argmax(preds, dim=1)

    #     return {
    #         'preds': preds,
    #         'targets': batch["semantic"],
    #         'image_id': batch['image_id']
    #     }
        
    
    def on_predict_epoch_end(self, results):
    #     #Save Panoptic results
        print("saving panoptic results")
        panoptic_predictions(self.cfg, results[0])
    #     print("saving panoptic results")
    #     semantic_predictions(self.cfg, results[0])
        

    def configure_optimizers(self):
        print("Optimizer - using {} with lr {}".format(self.cfg.SOLVER.NAME, self.learning_rate))
        if self.cfg.SOLVER.NAME == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.learning_rate,
                                         weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        elif self.cfg.SOLVER.NAME == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.learning_rate,
                                        momentum=0.9,
                                        weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        else:
            raise ValueError("Solver name is not supported, \
                Adam or SGD : {}".format(self.cfg.SOLVER.NAME))
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': ReduceLROnPlateau(self.optimizer,
                                              mode='min',
                                              patience=5,
                                              factor=0.1,
                                              min_lr=self.cfg.SOLVER.BASE_LR_PAN_DEPTH*1e-4,
                                              verbose=True),
            'monitor': 'train_loss_epoch'
        }

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.trainer.global_step < self.cfg.SOLVER.WARMUP_ITERS*self.cfg.NUM_GPUS:
            lr_scale = min(1., float(self.trainer.global_step + 1) /
                                    float(self.cfg.SOLVER.WARMUP_ITERS*self.cfg.NUM_GPUS))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.learning_rate

        # update params
        optimizer.step(closure=closure)
