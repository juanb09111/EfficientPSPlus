import sys
import os.path
from PIL import Image
import numpy as np
from pycocotools import mask as coco_mask
import json
from detectron2.config import get_cfg
from utils.add_custom_params import add_custom_params
from utils.get_vkitti_files import get_vkitti_files
from datasets.vkitti_cats import rgb_2_class, categories
# from pathlib import Path



def get_img_obj(arg):
    im_id, img_filename = arg
    image = Image.open(img_filename)
    width, height = image.size
    file_name = "/".join(img_filename.split("/")[-7:])
    obj = {
        "id": im_id,
        "width": width,
        "height": height,
        "file_name": file_name
    }

    return obj


# def map_semseg_img(img, basename, semantic_map_dest):
#     h, w = img.shape[:2]
#     new_img_tensor = np.zeros([h, w])
#     for i in range(h):
#         for j in range(w):
#             rgb = np.asarray(img[i, j])
#             if len(np.where((rgb != [0, 0, 0]))[0]) > 0:
#                 pix_cat = list(filter(lambda rgb2class_tup: (
#                     rgb2class_tup[1] == rgb).all(), rgb_2_class))[0]
#                 pix_cat = pix_cat[3]
#                 new_img_tensor[i, j] = pix_cat

#     I8 = (((new_img_tensor))).astype(np.uint8)
#     im = Image.fromarray(I8)

#     folder_path = os.path.join(semantic_map_dest)
#     Path(folder_path).mkdir(parents=True, exist_ok=True)
#     filename = os.path.join(folder_path, "classgt_" + basename + ".png")
#     im.save(filename)

#     return filename


# def vkitti2coco(cfg, rgb_root, instance_seg_root, semantic_seg_root, semantic_map_folder):
def vkitti2coco():
    cfg = get_cfg()
    add_custom_params(cfg)
    cfg.merge_from_file("configs/effps.yml")

    # sys.stdout = open(cfg.OUTPUT_FILE, 'w+')

    data = {}

    data["categories"] = categories
    data["annotations"] = []
    
    exclude = cfg.VKITTI_DATASET.EXCLUDE
    root = cfg.VKITTI_DATASET.DATASET_PATH.ROOT
    rgb_root = os.path.join(root, cfg.VKITTI_DATASET.DATASET_PATH.RGB)
    semantic_root = os.path.join(root, cfg.VKITTI_DATASET.DATASET_PATH.SEMANTIC)
    instance_root = os.path.join(root, cfg.VKITTI_DATASET.DATASET_PATH.INSTANCE)

    images = sorted(get_vkitti_files(rgb_root, exclude, "jpg"))

    print("Found {} images".format(len(images)))

    
    image_list = list(map(get_img_obj, list(enumerate(images))))

    data["images"] = image_list

    # -----instance and semantic seg images----------
    instance_seg_files = get_vkitti_files(instance_root, exclude, "png")
    semantic_seg_files = get_vkitti_files(semantic_root, exclude, "png")


    for img in data["images"]:
        
        img_filename = img["file_name"]
        # find scene and basename
        scene = img_filename.split("/")[-6]
        basename = img_filename.split(".")[-2].split("_")[-1]

        # Find corresponding semantic and instance images
        instance_img_filename = [s for s in instance_seg_files if (
            scene in s and basename in s)][0]
        semantic_img_filename = [s for s in semantic_seg_files if (
            scene in s and basename in s)][0]

        if instance_img_filename is None or semantic_img_filename is None:
            print("Warning! Instance image: {} Semantic image: {}".format(
                instance_img_filename, semantic_img_filename))

#         # Copy folder struct

#         folder_tree_init = semantic_img_filename.split(
#             "/").index(config_kitti.SEMANTIC_SEGMENTATION_DATA)

#         folder_tree = "/".join(semantic_img_filename.split("/")[folder_tree_init+1:-1])

        instance_img = Image.open(instance_img_filename)
        instance_img_arr = np.asarray(instance_img)
        unique_values = np.unique(instance_img_arr)
        unique_values = np.delete(unique_values, np.where(unique_values == 0))

        semseg_img = Image.open(semantic_img_filename)
        semseg_img_arr = np.asarray(semseg_img)

#         semantic_map_dest = os.path.join(os.path.dirname(os.path.abspath(
#             __file__)), semantic_map_folder, folder_tree)

        
#         semseg_filename = map_semseg_img(semseg_img_arr, basename, semantic_map_dest)

        
#         sem_loc = semseg_filename.split("/").index(semantic_map_folder.split("/")[-1])
#         sem_loc = "/".join(semseg_filename.split("/")[sem_loc:])
        
#         data["images"][idx]["semseg_img_filename"] = sem_loc

#         u_values = np.unique(semseg_img_arr)
#         u_values = np.delete(u_values, np.where(u_values == 0))
#         # print(u_values)
        for instance in unique_values:
#             # find coors where pixels equals current instance val
            coors = np.where(instance_img_arr == instance)
            coors_zip = list(zip(coors[0], coors[1]))

            # use ONE of those coordinates to find the class from the semantic seg image
            sample_coor = coors_zip[0]

            rgb = semseg_img_arr[sample_coor]
            cat = list(filter(lambda rgb2class_tup: (
                rgb2class_tup[1] == rgb).all(), rgb_2_class))[0]
            cat_name = cat[0]
            # generate mask and convert it to coco using pycoco tools

            mask = np.zeros_like(instance_img_arr)

            mask[coors[0], coors[1]] = 1

            encoded_mask = coco_mask.encode(np.asfortranarray(mask))
            bbx = coco_mask.toBbox(encoded_mask)
            area = coco_mask.area(encoded_mask)

            encoded_mask['counts'] = encoded_mask['counts'].decode("utf-8")

            category_id = list(
                filter(lambda category: category["name"] == cat_name, categories))[0]["id"]

            annotation_obj = {
                "id": len(data["annotations"]),
                "image_id": img["id"],
                "category_id": category_id,
                "segmentation": encoded_mask,
                "area": int(area),
                "bbox": list(bbx),
                "iscrowd": 0
            }

            data["annotations"].append(annotation_obj)

            # break
        # break
    dst = os.path.join(cfg.VKITTI_DATASET.DATASET_PATH.ROOT, cfg.VKITTI_DATASET.DATASET_PATH.COCO_ANNOTATION)
    with open(dst, 'w') as outfile:
        json.dump(data, outfile)