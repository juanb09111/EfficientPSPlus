import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from .fpn import TwoWayFpn
import pytorch_lightning as pl
from .backbone import generate_backbone_EfficientPS, output_feature_size
from .semantic_head import SemanticHead
from .instance_head import InstanceHead
# from .visualize_pred import visualize_pred
from .panoptic_segmentation_module import panoptic_segmentation_module
from .panoptic_metrics import generate_pred_panoptic
from .panoptic_predictions import panoptic_predictions
from panopticapi.evaluation import pq_compute
import os.path
# import cv2
# import matplotlib.pyplot as plt

class EffificientPS(pl.LightningModule):
    """
    EfficientPS model see http://panoptic.cs.uni-freiburg.de/
    Here pytorch lightningis used https://pytorch-lightning.readthedocs.io/en/latest/
    """

    def __init__(self, cfg):
        """
        Args:
        - cfg (Config) : Config object from detectron2
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = cfg.SOLVER.BASE_LR
        self.cfg = cfg
        self.backbone = generate_backbone_EfficientPS(cfg)
        self.fpn = TwoWayFpn(
            output_feature_size[cfg.MODEL_CUSTOM.BACKBONE.EFFICIENTNET_ID])
        self.semantic_head = SemanticHead(cfg.NUM_CLASS)
        self.instance_head = InstanceHead(cfg)
        
        # self.epoch = 0
    
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
        [self.log(k, v, batch_size=self.cfg.BATCH_SIZE) for k,v in loss.items()]
        self.log('train_loss', sum(loss.values()), batch_size=self.cfg.BATCH_SIZE)
        return {'loss': sum(loss.values())}

    def shared_step(self, inputs):
        loss = dict()
        predictions = dict()
        # Feature extraction
        features = self.backbone.extract_endpoints(inputs['image'])
        pyramid_features = self.fpn(features)
        # Heads Predictions
        output_size = inputs["image"][0].shape[-2:]
        semantic_logits, semantic_loss = self.semantic_head(pyramid_features, output_size, inputs)
        pred_instance, instance_losses = self.instance_head(pyramid_features, inputs)
        # Output set up
        loss.update(semantic_loss)
        loss.update(instance_losses)
        predictions.update({'semantic': semantic_logits})
        predictions.update({'instance': pred_instance})
        return predictions, loss

    def validation_step(self, batch, batch_idx):

        
        predictions, loss = self.shared_step(batch)
        panoptic_result = panoptic_segmentation_module(self.cfg,
            predictions,
            self.device)
        
        self.log("semantic_loss", loss["semantic_loss"], batch_size=self.cfg.BATCH_SIZE, sync_dist=True)
        self.log("val_loss", sum(loss.values()), batch_size=self.cfg.BATCH_SIZE, sync_dist=True)
        return {
            'val_loss': sum(loss.values()),
            'panoptic': panoptic_result,
            'image_id': batch['image_id']
        }
        

    def validation_epoch_end(self, outputs):
        # print(outputs)
        if self.local_rank == 0:

            gathered_results = self.all_gather(outputs)

           
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
                gt_folder= os.path.join(
                    self.cfg.VKITTI_DATASET.DATASET_PATH.ROOT,
                    self.cfg.VKITTI_DATASET.DATASET_PATH.COCO_PANOPTIC_SEGMENTATION),
                pred_folder=os.path.join(
                    self.cfg.VKITTI_DATASET.DATASET_PATH.ROOT,
                    self.cfg.VKITTI_DATASET.DATASET_PATH.PRED_DIR)
            )
            self.log("PQ", 100 * pq_res["All"]["pq"])
            self.log("SQ", 100 * pq_res["All"]["sq"])
            self.log("RQ", 100 * pq_res["All"]["rq"])
            self.log("PQ_th", 100 * pq_res["Things"]["pq"])
            self.log("SQ_th", 100 * pq_res["Things"]["sq"])
            self.log("RQ_th", 100 * pq_res["Things"]["rq"])
            self.log("PQ_st", 100 * pq_res["Stuff"]["pq"])
            self.log("SQ_st", 100 * pq_res["Stuff"]["sq"])
            self.log("RQ_st", 100 * pq_res["Stuff"]["rq"])
        
        else:
            return 0

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        predictions = dict()
        # Feature extraction
        features = self.backbone.extract_endpoints(batch['image'])
        pyramid_features = self.fpn(features)
        # Heads Predictions
        output_size = batch["image"][0].shape[-2:]
        semantic_logits, _ = self.semantic_head(pyramid_features, output_size)
        pred_instance, _ = self.instance_head(pyramid_features)

        predictions.update({'semantic': semantic_logits})
        predictions.update({'instance': pred_instance})

        #Panoptic fusion
        panoptic_result = panoptic_segmentation_module(self.cfg,
            predictions,
            self.device)

        return {'panoptic': panoptic_result,
            'image_id': batch['image_id']
        }
        
    
    def on_predict_epoch_end(self, results):
        #Save Panoptic results
        print("saving panoptic results")
        panoptic_predictions(self.cfg, results[0])
        

    def configure_optimizers(self):
        print("Optimizer - using {} with lr {}".format(self.cfg.SOLVER.NAME, self.cfg.SOLVER.BASE_LR))
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
                                              mode='max',
                                              patience=10,
                                              factor=0.1,
                                              min_lr=self.cfg.SOLVER.BASE_LR*1e-4,
                                              verbose=True),
            'monitor': 'PQ'
        }

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.trainer.global_step < self.cfg.SOLVER.WARMUP_ITERS:
            lr_scale = min(1., float(self.trainer.global_step + 1) /
                                    float(self.cfg.SOLVER.WARMUP_ITERS))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.cfg.SOLVER.BASE_LR

        # update params
        optimizer.step(closure=closure)
