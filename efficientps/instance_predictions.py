import torch
import numpy as np
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BitMasks
import os.path
from tqdm import tqdm
import cv2
import torch
import torch.nn.functional as F
from .panoptic_segmentation_module import check_bbox_size, scale_resize_pad_masks

def instance_predictions(cfg, outputs):

    # Create prediction dir if needed
    if cfg.DATASET_TYPE == "vkitti2":
        pred_dir = os.path.join(cfg.VKITTI_DATASET.DATASET_PATH.ROOT, cfg.VKITTI_DATASET.DATASET_PATH.PRED_DIR_INSTANCE)
        if not os.path.exists(pred_dir): os.makedirs(pred_dir)
    elif cfg.DATASET_TYPE == "odFridgeObjects":
        pred_dir = "tb_logs_2/maskrcnn/preds_2"
        if not os.path.exists(pred_dir): os.makedirs(pred_dir)
    elif cfg.DATASET_TYPE == "yt":
        pred_dir = os.path.join(cfg.YT_DATASET.DATASET_PATH.ROOT, cfg.YT_DATASET.DATASET_PATH.PRED_DIR_INSTANCE)
        if not os.path.exists(pred_dir): os.makedirs(pred_dir)
    elif cfg.DATASET_TYPE == "forest":
        pred_dir = os.path.join(cfg.FOREST_DATASET.DATASET_PATH.ROOT, cfg.FOREST_DATASET.DATASET_PATH.PRED_DIR_INSTANCE)
        if not os.path.exists(pred_dir): os.makedirs(pred_dir)

    for output in tqdm(outputs):
        if output["preds"] != None:
            for image, preds, targets, image_id in zip(output['images'], output['preds'], output['targets'], output['image_id']):
                image_id = image_id if isinstance(image_id, str) else image_id.item()
                im = image.cpu().numpy().transpose((1, 2, 0))*255
                vis = Visualizer(im)

                instance = check_bbox_size(preds)
                if instance.has('pred_masks'):
                    masks = scale_resize_pad_masks(instance)
                    # p2d = (0, 0, 1, 1)
                    # masks = F.pad(masks, p2d, "constant", 0)
                    masks =np.asarray([mask.cpu().numpy() for mask in masks])
                    masks = np.where(masks > 0.5, int(1), 0)
                    instance.pred_masks = masks
                vis.draw_instance_predictions(instance)
                output = vis.get_output()


                file_name = os.path.join(pred_dir, "{}_preds_instance.png".format(image_id))
                output.save(file_name)
                # cv2.imwrite(file_name, output)
                # print(output)
                    

        else:
            # for image, image_id in zip(output['images'], output['image_id']):
            #     image_id = image_id if isinstance(image_id, str) else image_id.item()
            #     im = image.cpu().numpy().transpose((1, 2, 0))*255
            #     file_name = os.path.join(pred_dir, "{}_preds_instance_no_detections.png".format(image_id))
            #     cv2.imwrite(file_name, im)
            print("no instances detected")