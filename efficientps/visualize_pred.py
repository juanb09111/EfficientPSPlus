import torch
import torch.nn.functional as F
from detectron2.structures import Instances
import os.path
import matplotlib.pyplot as plt
import cv2
from .panoptic_segmentation_module import check_bbox_size, scale_resize_pad
from utils.show_ann import visualize_masks, visualize_bboxes

def visualize_pred(cfg, imgs, outputs, device, batch_idx):
    """
    
    Visualize instance predictions and semantic predictions

    Args:
    - cfg (Config) : Config object
    - outputs (dict) : Inference output of our model
    - device : Device used by the lightning module

    Returns:
    - canvas (tensor) : [B, H, W] Panoptic predictions
    """
    #1 Source images
    visualize_imgs(cfg, imgs, batch_idx)
    #2 Semantic results 
    compute_output_only_semantic(cfg, outputs['semantic'], batch_idx)
    if outputs['instance'] is not None:
        #3 Visualize instance
        visualize_instance_pred(cfg, outputs["instance"], batch_idx, device)
    

def visualize_imgs(cfg, imgs, batch_idx):

    for idx, im in enumerate(imgs):
        image = im.cpu().numpy().transpose((1, 2, 0))*255
        cv2.imwrite(os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR, "inference_test", "img_batch_idx_{}_idx_{}_.png".format(batch_idx, idx)), image)

def compute_output_only_semantic(cfg, semantic, batch_idx):
    """
    Visualize semantic outputs.
    
    Args:
    - semantic (tensor) : Output of the semantic head (either for the full
                          batch or for one image)
    """
    for i, sem in enumerate(semantic):
        plt.imshow(torch.argmax(sem, dim=0).cpu().numpy())
        plt.savefig(os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR, "inference_test", "semantic_output_batch_idx_{}_idx_{}.png".format(batch_idx, i)))

def visualize_instance_pred(cfg, instance_pred, batch_idx, device):

    for idx, pred in enumerate(instance_pred):
        instance = check_bbox_size(pred)
        if len(instance.pred_boxes.tensor) > 0:
            masks = scale_resize_pad(instance).to(device)
            masks = torch.where(masks > 0.5, int(1), 0)
            # print(masks.shape)
            boxes = instance.pred_boxes.tensor

            image = visualize_masks(masks)
            image = visualize_bboxes(image, boxes)
            cv2.imwrite(os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR, "inference_test", "masks_output_batch_idx_{}_idx_{}.png".format(batch_idx, idx)), image)
            



