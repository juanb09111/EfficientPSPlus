from operator import length_hint
import os
import json
from tqdm import tqdm
import numpy as np
from detectron2.utils.visualizer import Visualizer
from PIL import Image
import torch.nn.functional as F
from panopticapi.utils import id2rgb
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from .panoptic_segmentation_module import check_bbox_size, scale_resize_pad_masks


import random
def randRGB(seed=None):
    if seed is not None:
        random.seed(seed)
    r = random.random()*255
    g = random.random()*255
    b = random.random()*255
    rgb = [int(r), int(g), int(b)]
    return rgb


def save_fig(im, loc, file_name, shape):

    height, width = shape
    # im = im.cpu().permute(1, 2, 0).numpy()

    dppi = 96
    fig, ax = plt.subplots(1, 1, figsize=(
        width/dppi, height/dppi), dpi=dppi)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    ax.imshow(im,  interpolation='nearest', aspect='auto')
    plt.axis('off')
    fig.savefig(os.path.join('{}/{}.png'.format(loc, file_name)))
    plt.close(fig)

def depth_predictions(cfg, outputs):
    """
    Take all output of a model and save a json file with al predictions as
    well as all panoptic image prediction.
    This is done in order to use `pq_compute` function from panoptic api
    Args:
    - cfg (Config) : config object
    - outputs (list[dict]) : List of a full epoch of outputs
    """
    # Create prediction dir if needed
    if cfg.DATASET_TYPE == "vkitti2":
        pred_dir = os.path.join(cfg.VKITTI_DATASET.DATASET_PATH.ROOT, cfg.VKITTI_DATASET.DATASET_PATH.PRED_DIR)
        if not os.path.exists(pred_dir): os.makedirs(pred_dir)
    elif cfg.DATASET_TYPE == "forest":
        pred_dir = os.path.join(cfg.FOREST_DATASET.DATASET_PATH.ROOT, cfg.FOREST_DATASET.DATASET_PATH.PRED_DIR_DEPTH)
        if not os.path.exists(pred_dir): os.makedirs(pred_dir)

    print("Saving depth predictions to ", pred_dir)
    # Loop on each validation output
    # print("outputs", len(outputs))
    for output in tqdm(outputs):
        # print("output", len(output), len(output[0]))
        # Loop on each image of the batch

        for idx, image_id in enumerate(output['image_id']):
            img_data = dict()
            img_data['image_id'] = image_id.item()
            
            #Save Depth
            if "depth" in output.keys():
                filename_depth = "{}_preds_depth_gray.png".format(img_data['image_id'])
                depth = output["depth"][idx]
                save_image(depth, os.path.join(pred_dir, filename_depth), normalize=True)
                depth = depth.squeeze(0)
                shape = depth.shape
                out_depth_numpy = depth.cpu().numpy()/255
                filename_depth = "{}_preds_depth".format(img_data['image_id'])
                save_fig(out_depth_numpy, pred_dir, filename_depth, shape)
            if "sparse_depth_gt" in output.keys():
                filename_depth = "{}_preds_sparse_depth_gt_gray.png".format(img_data['image_id'])
                depth = output["sparse_depth_gt"][idx]
                save_image(depth, os.path.join(pred_dir, filename_depth), normalize=True)
                depth = depth.squeeze(0)
                shape = depth.shape
                out_depth_numpy = depth.cpu().numpy()/255
                filename_depth = "{}_preds_sparse_depth_gt".format(img_data['image_id'])
                save_fig(out_depth_numpy, pred_dir, filename_depth, shape)
            if "rgb_image" in output.keys():
                filename_depth = "{}_preds_rgb.png".format(img_data['image_id'])
                rgb_im = output["rgb_image"][idx]
                save_image(rgb_im, os.path.join(pred_dir, filename_depth))


