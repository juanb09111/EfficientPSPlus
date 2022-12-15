#!/usr/bin/env python
'''
This script converts detection COCO format to panoptic COCO format. More
information about the formats can be found here:
http://cocodataset.org/#format-data.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import os, sys
import argparse
import numpy as np
import json
import time
import multiprocessing
from detectron2.config import get_cfg
from add_custom_params import add_custom_params
import os.path
import shutil
import os

import PIL.Image as Image

from panopticapi.utils import rgb2id
from get_forest_files import get_forest_files

def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

mapping = {
    (147,228,202): 1,
    (65,20,19): 2,
    (64,0,160): 3,
    (57,240,237): 4,
    (144,96,0): 5,
    (128,10,129): 6,
    (130,44,136): 7,
    (104,173,142): 8,
    (0, 0, 0): 0
}



def mask_to_class(mask):
        target = torch.from_numpy(mask)
        h,w = target.shape[0],target.shape[1]
        masks = torch.empty(h, w, dtype=torch.long)
        target = target.permute(2, 0, 1).contiguous()
        for k in mapping:
            idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3) 
            masks[validx] = torch.tensor(mapping[k], dtype=torch.long)
        I8 = (((masks.cpu().numpy()))).astype(np.uint8)
        im = Image.fromarray(I8)
        return im

def generate_pan_2ch(args):
    
    cfg = get_cfg()
    add_custom_params(cfg)
    cfg.merge_from_file(args.config)


    image_json = os.path.join("..", cfg.FOREST_DATASET.DATASET_PATH.ROOT, cfg.FOREST_DATASET.DATASET_PATH.COCO_ANNOTATION_TRAIN)
    semantic_folder = os.path.join("..", cfg.FOREST_DATASET.DATASET_PATH.ROOT, cfg.FOREST_DATASET.DATASET_PATH.SEMANTIC_TRAIN)
    instance_folder = os.path.join("..", cfg.FOREST_DATASET.DATASET_PATH.ROOT, cfg.FOREST_DATASET.DATASET_PATH.INSTANCE_TRAIN)
    two_ch_folder = os.path.join("..", cfg.FOREST_DATASET.DATASET_PATH.ROOT, cfg.FOREST_DATASET.DATASET_PATH.TWO_CH_PANOPTIC_SEGMENTATION_TRAIN)
    kitti_2ch_panoptic_json = os.path.join("..", cfg.FOREST_DATASET.DATASET_PATH.ROOT, cfg.FOREST_DATASET.DATASET_PATH.TWO_CH_IMAGE_JSON_TRAIN)

    # image_json = os.path.join("..", cfg.FOREST_DATASET.DATASET_PATH.ROOT, cfg.FOREST_DATASET.DATASET_PATH.COCO_ANNOTATION_VAL)
    # semantic_folder = os.path.join("..", cfg.FOREST_DATASET.DATASET_PATH.ROOT, cfg.FOREST_DATASET.DATASET_PATH.SEMANTIC_VAL)
    # instance_folder = os.path.join("..", cfg.FOREST_DATASET.DATASET_PATH.ROOT, cfg.FOREST_DATASET.DATASET_PATH.INSTANCE_VAL)
    # two_ch_folder = os.path.join("..", cfg.FOREST_DATASET.DATASET_PATH.ROOT, cfg.FOREST_DATASET.DATASET_PATH.TWO_CH_PANOPTIC_SEGMENTATION_VAL)
    # kitti_2ch_panoptic_json = os.path.join("..", cfg.FOREST_DATASET.DATASET_PATH.ROOT, cfg.FOREST_DATASET.DATASET_PATH.TWO_CH_IMAGE_JSON_VAL)


    semantic_imgs = get_forest_files(semantic_folder, "png")
    instance_imgs = get_forest_files(instance_folder, "png")

    with open(image_json, 'r') as f:
        data = json.load(f)

    image_data = data["images"]

    two_ch_images = []

    for image in image_data:
        
        
        file_name = image["file_name"]
        file_name = ".".join([*image['file_name'].rsplit('.')[:-1], "png"])
        basename = file_name.split(".")[-2]

        semantic_img_filename = [s for s in semantic_imgs if basename in s][0]
        instance_img_filename = [s for s in instance_imgs if basename in s][0]

        semantic_mask = Image.open(semantic_img_filename)
        instance_mask = Image.open(instance_img_filename).convert('RGB')

        two_ch_pan = np.zeros_like(np.asarray(semantic_mask))
        #RGB to class
        semantic_mask = mask_to_class(np.asarray(semantic_mask))
        semantic_mask = np.asarray(semantic_mask, dtype=np.long)
        
        instance_mask = rgb2id(np.asarray(instance_mask))

        id_mask = np.zeros_like(instance_mask)
        for idx, val in enumerate(np.unique(instance_mask)):
            if val != 0:
                id_mask[instance_mask == val] = idx +1

        # print(np.unique(semantic_mask), np.unique(instance_mask))
        two_ch_pan[:,:,0] = semantic_mask
        two_ch_pan[:,:,1] = id_mask
        file_name = os.path.join(two_ch_folder, file_name)
        
        Image.fromarray(two_ch_pan).save(file_name)

        json_file_name = "/".join(file_name.split("/")[-4:])
        two_ch_images.append({
            **image,
            "file_name": json_file_name
        })
    
    two_ch_json = {"images":two_ch_images, "categories": data["categories"]}

    with open(kitti_2ch_panoptic_json, 'w') as f:
        json.dump(two_ch_json, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script converts semantic and instance annotations to 2ch panoptic segmentation"
    )
    parser.add_argument('--config', type=str,
                        help="config yml location")
    
    args = parser.parse_args()

    generate_pan_2ch(args)