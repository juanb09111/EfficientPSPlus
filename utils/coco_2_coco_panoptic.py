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

from panopticapi.utils import get_traceback, IdGenerator, save_json


# defining the function to ignore the files
# if present in any folder
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

def save_json(d, file):
    with open(file, 'w') as f:
        json.dump(d, f)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

try:
    # set up path for pycocotools
    # sys.path.append('./cocoapi-master/PythonAPI/')
    from pycocotools import mask as COCOmask
    from pycocotools.coco import COCO as COCO
except Exception:
    raise Exception("Please install pycocotools module from https://github.com/cocodataset/cocoapi")

@get_traceback
def convert_detection_to_panoptic_coco_format_single_core(
    proc_id, coco_detection, img_ids, categories, segmentations_folder
):
    id_generator = IdGenerator(categories)
    
    annotations_panoptic = []
    for working_idx, img_id in enumerate(img_ids):
        if working_idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id,
                                                                 working_idx,
                                                                 len(img_ids)))
        img = coco_detection.loadImgs(int(img_id))[0]
        pan_format = np.zeros((img['height'], img['width'], 3), dtype=np.uint8)
        overlaps_map = np.zeros((img['height'], img['width']), dtype=np.uint32)

        anns_ids = coco_detection.getAnnIds(img_id)
        anns = coco_detection.loadAnns(anns_ids)

        panoptic_record = {}
        panoptic_record['image_id'] = img_id
        file_name = ".".join([*img['file_name'].rsplit('.')[:-1], "png"])
        file_name = os.path.join(segmentations_folder.split("/")[-1], "/".join(file_name.split("/")[1:]))
        # print(file_name)
        panoptic_record['file_name'] = file_name
        segments_info = []
        for ann in anns:
            if ann['category_id'] not in categories:
                raise Exception('Panoptic coco categories file does not contain \
                    category with id: {}'.format(ann['category_id'])
                )
            segment_id, color = id_generator.get_id_and_color(ann['category_id'])
            mask = coco_detection.annToMask(ann)
            overlaps_map += mask
            pan_format[mask == 1] = color
            ann.pop('segmentation')
            ann.pop('image_id')
            ann['id'] = segment_id
            segments_info.append(ann)

        if np.sum(overlaps_map > 1) != 0:
            raise Exception("Segments for image {} overlap each other.".format(img_id))
        panoptic_record['segments_info'] = segments_info
        annotations_panoptic.append(panoptic_record)

        Image.fromarray(pan_format).save(os.path.join("/".join(segmentations_folder.split("/")[:-1]), file_name))

    print('Core: {}, all {} images processed'.format(proc_id, len(img_ids)))
    return annotations_panoptic


def convert_detection_to_panoptic_coco_format(input_json_file,
                                              segmentations_folder,
                                              output_json_file,
                                              categories_json_file):
    start_time = time.time()

    # if segmentations_folder is None:
    #     segmentations_folder = output_json_file.rsplit('.', 1)[0]
    # if not os.path.isdir(segmentations_folder):
    #     print("Creating folder {} for panoptic segmentation PNGs".format(segmentations_folder))
    #     os.mkdir(segmentations_folder)

    print("CONVERTING...")
    print("COCO detection format:")
    print("\tJSON file: {}".format(input_json_file))
    print("TO")
    print("COCO panoptic format")
    print("\tSegmentation folder: {}".format(segmentations_folder))
    print("\tJSON file: {}".format(output_json_file))
    print('\n')
    print(input_json_file)
    coco_detection = COCO(input_json_file)
    img_ids = coco_detection.getImgIds()
    categories_list = coco_detection.cats.values()
    categories_list = list(map(lambda cat: {**cat, 
        "isthing": cat["supercategory"] == "object",
        "color": rgb_2_class[cat["id"] - 1][1]
        }, categories_list))
    # print(coco_detection.cats.values())
    # with open(categories_json_file, 'r') as f:
    #     categories_list = json.load(f)
    categories = {category['id']: category for category in categories_list}

    cpu_num = multiprocessing.cpu_count()
    img_ids_split = np.array_split(img_ids, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(img_ids_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, img_ids in enumerate(img_ids_split):
        p = workers.apply_async(convert_detection_to_panoptic_coco_format_single_core,
                                (proc_id, coco_detection, img_ids, categories, segmentations_folder))
        processes.append(p)
    annotations_coco_panoptic = []
    for p in processes:
        annotations_coco_panoptic.extend(p.get())

    with open(input_json_file, 'r') as f:
        d_coco = json.load(f)
    d_coco['annotations'] = annotations_coco_panoptic
    d_coco['categories'] = categories_list

    # cls=NpEncoder
    # save_json(d_coco, output_json_file)

    with open(output_json_file, 'w') as f:
        json.dump(d_coco, f, cls=NpEncoder)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))


def coco2coco_panoptic(args):

    cfg = get_cfg()
    add_custom_params(cfg)
    cfg.merge_from_file(args.config)

    input_json = os.path.join("..", cfg.VKITTI_DATASET.DATASET_PATH.ROOT, cfg.VKITTI_DATASET.DATASET_PATH.COCO_ANNOTATION)
    output_json_file = os.path.join("..", cfg.VKITTI_DATASET.DATASET_PATH.ROOT, cfg.VKITTI_DATASET.DATASET_PATH.COCO_PANOPTIC_ANNOTATION)
    panoptic_coco_categories = os.path.join("..", cfg.VKITTI_DATASET.DATASET_PATH.ROOT, "panoptic_coco_categories.json")
    segmentations_folder = os.path.join("..", cfg.VKITTI_DATASET.DATASET_PATH.ROOT, cfg.VKITTI_DATASET.DATASET_PATH.COCO_PANOPTIC_SEGMENTATION)
    # if not os.path.isdir(segmentations_folder):
    #     print("Creating folder {} for panoptic segmentation PNGs".format(segmentations_folder))
    #     os.mkdir(segmentations_folder)
    #copy folder structure
    if not os.path.isdir(segmentations_folder):
        src_folder = input_json = os.path.join("..", cfg.VKITTI_DATASET.DATASET_PATH.ROOT, cfg.VKITTI_DATASET.DATASET_PATH.RGB)
        shutil.copytree(src_folder,
                    segmentations_folder,
                    ignore=ignore_files)
    convert_detection_to_panoptic_coco_format(input_json, segmentations_folder, output_json_file, panoptic_coco_categories)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script converts detection COCO format to panoptic \
            COCO format. See this file's head for more information."
    )
    parser.add_argument('--config', type=str,
                        help="config yml location")
    
    args = parser.parse_args()

    coco2coco_panoptic(args)