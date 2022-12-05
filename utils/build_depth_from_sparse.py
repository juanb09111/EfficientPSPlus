
import os
import os.path
import torch
from PIL import Image
import numpy as np
import random
import time
import argparse
import multiprocessing
from detectron2.config import get_cfg
from panopticapi.utils import get_traceback
from add_custom_params import add_custom_params
from get_forest_files import get_forest_files
import shutil
from tqdm import tqdm
from multiprocessing import  RLock

def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]


@get_traceback
def build_single_core(cfg, proc_id, image_set, virtual_gt_folder, depth_proj_folder):

    tqdm_text = "#" + "{}".format(proc_id).zfill(3)
    with tqdm(total=len(image_set), desc=tqdm_text, position=proc_id+1) as pbar:

        for working_idx, file_name in enumerate(image_set):
            
            # if working_idx % 100 == 0:
            #     print('Core: {}, {} from {} images converted'.format(proc_id, working_idx, len(image_set)))
            
            dst_file_name = file_name.split("/")[-1]
            file_path_proj = os.path.join(depth_proj_folder, dst_file_name)
            
            if os.path.isfile(file_path_proj):
                
                continue

            depth_img = np.asarray(Image.open(file_name))
            
            img_height, img_width  = depth_img.shape
            
            N_num_training = np.floor(img_height*img_width*cfg.FOREST_DATASET.DEPTH.SPARSITY_TRAINING)

            # Find depth values within range
            depth = torch.tensor(depth_img)
            coors = torch.nonzero(depth)
            random.Random(4).shuffle(coors)
            
            depth_ = depth[coors[:, 0], coors[:, 1]]/255
            inds = (depth_ > 0)
            
            
            # Synthetic lidar
            if not os.path.isfile(file_path_proj):

                coors_lidar = coors[inds, :][:int(N_num_training)]
                
                depth_lidar = depth[coors_lidar[:, 0], coors_lidar[:, 1]][:int(N_num_training)]

                sparse_depth = torch.zeros_like(depth)
                sparse_depth[coors_lidar[:, 0], coors_lidar[:, 1]] = depth_lidar

            
                im = Image.fromarray(sparse_depth.numpy())
                im.save(file_path_proj)

            # depth_img_ = np.asarray(Image.open(file_path_proj))
            # print("res", np.unique(depth_img_/255), np.count_nonzero(depth_img_))
            # print(file_path_gt)
            pbar.update(1)
    
    return 0
            

def build_depth_dataset(args):

    start_time = time.time()

    cfg = get_cfg()
    add_custom_params(cfg)
    cfg.merge_from_file(args.config)

    root = cfg.FOREST_DATASET.DATASET_PATH.ROOT

    gt_folder = os.path.join("..", root, cfg.FOREST_DATASET.DATASET_PATH.DEPTH_TRAIN)
    depth_proj_folder = os.path.join("..", root, cfg.FOREST_DATASET.DATASET_PATH.DEPTH_PROJ_TRAIN)

    depth_imgs = get_forest_files(gt_folder, "png")
    
    
    cpu_num = multiprocessing.cpu_count()
    images_split = np.array_split(depth_imgs, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(images_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num, initargs=(RLock(),), initializer=tqdm.set_lock)
    processes = []
    
    for proc_id, image_set in enumerate(images_split):
        p = workers.apply_async(build_single_core, (cfg, proc_id, image_set, gt_folder, depth_proj_folder))
        processes.append(p)
    
    # for p in tqdm(processes):
    #     p.get()
    workers.close()
    [p.get() for p in processes]
    print("\n" * (len(images_split) + 1))
    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script converts semantic and instance annotations to 2ch panoptic segmentation"
    )
    parser.add_argument('--config', type=str,
                        help="config yml location")
    
    args = parser.parse_args()

    build_depth_dataset(args)

