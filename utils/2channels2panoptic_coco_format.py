#!/usr/bin/env python
'''
This script converts panoptic segmentation predictions stored in 2 channels
panoptic format to COCO panoptic format.
2 channels format is described in the panoptic segmentation paper
(https://arxiv.org/pdf/1801.00868.pdf). Two labels are assigned to each pixel of
a segment:
- semantic class label;
- instance ID (nonnegative integer).
PNG format is used to store the data. The first channel stores semantic class
of a pixel and the second one stores instance ID.
For stuff categories instance ID is redundant and is 0 for all pixels
corresponding to stuff segments.
Panoptic COCO format is described fully in http://cocodataset.org/#format-data.
It is used for the Panoptic COCO challenge evaluation.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import argparse
import numpy as np
import json
import time
import multiprocessing
import itertools
from detectron2.config import get_cfg
from add_custom_params import add_custom_params
import shutil

import PIL.Image as Image

from panopticapi.utils import get_traceback, IdGenerator, save_json

try:
    # set up path for pycocotools
    # sys.path.append('./cocoapi-master/PythonAPI/')
    from pycocotools import mask as COCOmask
    from pycocotools.coco import COCO as COCO
except Exception:
    raise Exception("Please install pycocotools module from https://github.com/cocodataset/cocoapi")

def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

rgb_2_class = [
    ("Terrain", [210, 0, 200], "background", 1),
    ("Sky", [90, 200, 255], "background", 2),
    ("Tree", [0, 199, 0], "background", 3),
    ("Vegetation", [90, 240, 0], "background", 4),
    ("Building", [140, 140, 140], "background", 5),
    ("Road", [100, 60, 100], "background", 6),
    ("GuardRail", [250, 100, 255], "background", 7),
    ("TrafficSign", [255, 255, 0], "background", 8),
    ("TrafficLight", [200, 200, 0], "background", 9),
    ("Pole", [255, 130, 0], "background", 10),
    ("Misc", [80, 80, 80], "background", 11),
    ("Truck", [160, 60, 60], "object", 12),
    ("Car", [255, 127, 80], "object", 13),
    ("Van", [0, 139, 139], "object", 14),
    ("Undefined", [0, 0, 0], "background", 0)
]

OFFSET = 1000

@get_traceback
def convert_single_core(proc_id, image_set, categories, source_folder, segmentations_folder, VOID=0):
    annotations = []
    for working_idx, image_info in enumerate(image_set):
        if working_idx % 100 == 0:
            print('Core: {}, {} from {} images converted'.format(proc_id, working_idx, len(image_set)))
        
        file_name = '{}.png'.format(image_info['file_name'].rsplit('.')[0])

        try:
            original_format = np.array(Image.open(os.path.join(source_folder, file_name)), dtype=np.uint32)
        except IOError:
            raise KeyError('no prediction png file for id: {}'.format(image_info['id']))

        pan = OFFSET * original_format[:, :, 0] + original_format[:, :, 1]
        pan_format = np.zeros((original_format.shape[0], original_format.shape[1], 3), dtype=np.uint8)

        id_generator = IdGenerator(categories)

        l, areas = np.unique(pan, return_counts=True)
        segm_info = []
        for el, area in zip(l, areas):
            sem = el // OFFSET
            if sem == VOID:
                continue
            if sem not in categories:
                raise KeyError('Unknown semantic label {}'.format(sem))
            mask = pan == el
            segment_id, color = id_generator.get_id_and_color(sem)
            pan_format[mask] = color
            segm_info.append({"id": segment_id,
            "iscrowd": 0,
            "area": area,
            "category_id": int(sem)})
        
        file_path = os.path.join(segmentations_folder, "/".join(file_name.split("/")[1:]))
        file_name = "/".join(file_path.split("/")[-7:])
        
        annotations.append({'image_id': image_info['id'],
                            'file_name': file_name,
                            "segments_info": segm_info})
        
        Image.fromarray(pan_format).save(os.path.join(file_path))
    print('Core: {}, all {} images processed'.format(proc_id, len(image_set)))
    return annotations


def converter(source_folder, images_json_file, categories_json_file,
              segmentations_folder, predictions_json_file,
              VOID=0):
    start_time = time.time()

    print("Reading image set information from {}".format(images_json_file))


    with open(images_json_file, 'r') as f:
        d_coco = json.load(f)
    images = d_coco['images']

    with open(categories_json_file, 'r') as f:
        kitti_coco = json.load(f)
    kitti_coco_images = kitti_coco['images']


    coco_detection = COCO(categories_json_file)
    categories_list = coco_detection.cats.values()
    categories_list = list(map(lambda cat: {**cat, 
        "isthing": 1 if cat["supercategory"] == "object" else 0,
        "color": rgb_2_class[cat["id"] - 1][1]
        }, categories_list))
    
    categories = {category['id']: category for category in categories_list}

    if segmentations_folder is None:
        segmentations_folder = predictions_json_file.rsplit('.', 1)[0]
    if not os.path.isdir(segmentations_folder):
        print("Creating folder {} for panoptic segmentation PNGs".format(segmentations_folder))
        os.mkdir(segmentations_folder)

    print("CONVERTING...")
    print("2 channels panoptic format:")
    print("\tSource folder: {}".format(source_folder))
    print("TO")
    print("COCO panoptic format:")
    print("\tSegmentation folder: {}".format(segmentations_folder))
    print("\tJSON file: {}".format(predictions_json_file))
    print('\n')
    cpu_num = multiprocessing.cpu_count()
    images_split = np.array_split(images, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(images_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, image_set in enumerate(images_split):
        p = workers.apply_async(convert_single_core,
                                (proc_id, image_set, categories, source_folder, segmentations_folder, VOID))
        processes.append(p)
    annotations = []
    for p in processes:
        annotations.extend(p.get())

    print("Writing final JSON in {}".format(predictions_json_file))
    d_coco['annotations'] = annotations
    d_coco['categories'] = categories_list
    d_coco['images'] = kitti_coco_images
    save_json(d_coco, predictions_json_file)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))


def convert(args):
    
    cfg = get_cfg()
    add_custom_params(cfg)
    cfg.merge_from_file(args.config)

    source_folder = os.path.join("..", cfg.VKITTI_DATASET.DATASET_PATH.ROOT)
    images_json_file = os.path.join("..", cfg.VKITTI_DATASET.DATASET_PATH.ROOT, cfg.VKITTI_DATASET.DATASET_PATH.TWO_CH_IMAGE_JSON)
    segmentations_folder = os.path.join("..", cfg.VKITTI_DATASET.DATASET_PATH.ROOT, cfg.VKITTI_DATASET.DATASET_PATH.COCO_PANOPTIC_SEGMENTATION)
    predictions_json_file = os.path.join("..", cfg.VKITTI_DATASET.DATASET_PATH.ROOT, cfg.VKITTI_DATASET.DATASET_PATH.COCO_PANOPTIC_ANNOTATION)
    categories_json_file = os.path.join("..", cfg.VKITTI_DATASET.DATASET_PATH.ROOT, cfg.VKITTI_DATASET.DATASET_PATH.COCO_ANNOTATION)

    if not os.path.isdir(segmentations_folder):
        src_folder = os.path.join("..", cfg.VKITTI_DATASET.DATASET_PATH.ROOT, cfg.VKITTI_DATASET.DATASET_PATH.RGB)
        print("copying folder structure from {} to {}".format(src_folder, segmentations_folder))
        shutil.copytree(src_folder,
                    segmentations_folder,
                    ignore=ignore_files)
    converter(source_folder, images_json_file, categories_json_file, segmentations_folder, predictions_json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script converts panoptic segmentation predictions \
        stored in 2 channels panoptic format to COCO panoptic format. See this \
        file's head for more information."
    )
    parser.add_argument('--config', type=str,
                        help="config yml location")
    
    args = parser.parse_args()

    convert(args)