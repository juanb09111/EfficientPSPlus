from operator import length_hint
import os
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch.nn.functional as F
from panopticapi.utils import id2rgb


def semantic_predictions(cfg, outputs):
    """
    Args:
    - cfg (Config) : config object
    - outputs (list[dict]) : List of a full epoch of outputs
    """
    # Create prediction dir if needed
    if cfg.DATASET_TYPE == "vkitti2":
        pred_dir = os.path.join(cfg.VKITTI_DATASET.DATASET_PATH.ROOT, cfg.VKITTI_DATASET.DATASET_PATH.PRED_DIR_SEMANTIC)
        if not os.path.exists(pred_dir): os.makedirs(pred_dir)
    elif cfg.DATASET_TYPE == "forest":
        pred_dir = os.path.join(cfg.FOREST_DATASET.DATASET_PATH.ROOT, cfg.FOREST_DATASET.DATASET_PATH.PRED_DIR_SEMANTIC)
        if not os.path.exists(pred_dir): os.makedirs(pred_dir)

    annotations = []
    print("Saving semantic predictions to ", pred_dir)
    # Loop on each validation output
    # print("outputs", len(outputs))
    for output in tqdm(outputs):
        # print("output", len(output), len(output[0]))
        # Loop on each image of the batch
        for img_semantic, image_id in zip(output['preds_sem'], output['image_id']):
            img_data = dict()
            img_data['image_id'] = image_id.item()
            # Resize the image to original size
            img_semantic = F.interpolate(
                img_semantic.unsqueeze(0).unsqueeze(0).float(),
                size=(cfg.VKITTI_DATASET.ORIGINAL_SIZE.HEIGHT, cfg.VKITTI_DATASET.ORIGINAL_SIZE.WIDTH),
                mode='nearest'
            )[0,0,...]
            # Create segment_info data
            img_data['segments_info'] = []
            img_semantic = img_semantic.cpu().numpy()
            instances, areas = np.unique(img_semantic, return_counts=True)
            for instance, area in zip(instances, areas):
                if instance == 0:
                    continue
                img_data['segments_info'].append(
                    {
                        'id': int(instance),
                        "area": int(area),
                        'category_id': int(instance)
                                       if instance < 1000
                                       else int(instance / 1000)
                    }
                )
                
            # Save semantic_preds
            img_data['file_name'] = "{}_preds_semantic.png".format(img_data['image_id'])
            img = id2rgb(img_semantic)
            img_to_save = Image.fromarray(img)
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
        
        pred_path = os.path.join(cfg.VKITTI_DATASET.DATASET_PATH.ROOT, cfg.VKITTI_DATASET.DATASET_PATH.PRED_JSON_SEMANTIC)
    elif cfg.DATASET_TYPE == "forest":
        pred_path = os.path.join(cfg.FOREST_DATASET.DATASET_PATH.ROOT, cfg.FOREST_DATASET.DATASET_PATH.PRED_JSON_SEMANTIC)

    json_data={}
    json_data['annotations'] = annotations
    with open(pred_path, "w") as f:
        f.write(json.dumps(json_data))