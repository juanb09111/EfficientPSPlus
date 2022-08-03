from operator import length_hint
import os
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch.nn.functional as F
from panopticapi.utils import id2rgb

import random
def randRGB(seed=None):
    if seed is not None:
        random.seed(seed)
    r = random.random()*255
    g = random.random()*255
    b = random.random()*255
    rgb = [int(r), int(g), int(b)]
    return rgb

def panoptic_predictions(cfg, outputs):
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

    annotations = []
    print("Saving panoptic predictions to ", pred_dir)
    # Loop on each validation output
    # print("outputs", len(outputs))
    for output in tqdm(outputs):
        # print("output", len(output), len(output[0]))
        # Loop on each image of the batch
        for img_panoptic, image_id in zip(output['panoptic'], output['image_id']):
            img_data = dict()
            img_data['image_id'] = image_id.item()
            # Resize the image to original size
            img_panoptic = F.interpolate(
                img_panoptic.unsqueeze(0).unsqueeze(0).float(),
                size=(cfg.VKITTI_DATASET.ORIGINAL_SIZE.HEIGHT, cfg.VKITTI_DATASET.ORIGINAL_SIZE.WIDTH),
                mode='nearest'
            )[0,0,...]
            # Create segment_info data
            img_data['segments_info'] = []
            img_panoptic = img_panoptic.cpu().numpy()
            rgb_shape = tuple(list(img_panoptic.shape) + [3])
            rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
            
            for instance in np.unique(img_panoptic):
                if instance == 0:
                    continue
                rgb = randRGB(int(instance))
                locs = np.where(img_panoptic == instance)
                rgb_map[locs] = rgb
                img_data['segments_info'].append(
                    {
                        'id': int(instance),
                        'category_id': int(instance)
                                       if instance < 1000
                                       else int(instance / 1000)
                    }
                )
            # Save panotic_pred
            img_data['file_name'] = "{}_preds_panoptic.png".format(img_data['image_id'])
            img = id2rgb(img_panoptic)
            img_to_save = Image.fromarray(rgb_map)
            img_to_save.save(os.path.join(pred_dir, img_data['file_name']))
            # Add annotation of a one image
            annotations.append(img_data)
    save_json_file(cfg, annotations)

def save_json_file(cfg, annotations):
    """
    Load gt json file to have same architecture and replace annotations
    with the prediction annotations

    Args:
    - cfg (Config) : config object
    - annotations (List[dict]) : List containing prediction info for each image
    """
    if cfg.DATASET_TYPE == "vkitti2":
        
        pred_path = os.path.join(cfg.VKITTI_DATASET.DATASET_PATH.ROOT, cfg.VKITTI_DATASET.DATASET_PATH.PRED_JSON)

    json_data={}
    json_data['annotations'] = annotations
    with open(pred_path, "w") as f:
        f.write(json.dumps(json_data))