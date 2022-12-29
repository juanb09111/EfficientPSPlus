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
    elif cfg.DATASET_TYPE == "forest":
        pred_dir = os.path.join(cfg.FOREST_DATASET.DATASET_PATH.ROOT, cfg.FOREST_DATASET.DATASET_PATH.PRED_DIR)
        if not os.path.exists(pred_dir): os.makedirs(pred_dir)

    annotations = []
    print("Saving panoptic predictions to ", pred_dir)
    # Loop on each validation output
    # print("outputs", len(outputs))
    for output in tqdm(outputs):
        # print("output", len(output), len(output[0]))
        # Loop on each image of the batch
        output_instances = [None for im in output['images']]
        if output["instance"] != None:
            output_instances = output["instance"]

        for idx, (rgb_img, img_panoptic, img_semantic, instance, image_id) in enumerate(zip(output['images'], output['panoptic'], output['semantic'], output_instances, output['image_id'])):
            img_data = dict()
            img_data['image_id'] = image_id.item()
            # Resize the image to original size
            # img_panoptic = F.interpolate(
            #     img_panoptic.unsqueeze(0).unsqueeze(0).float(),
            #     size=(cfg.VKITTI_DATASET.ORIGINAL_SIZE.HEIGHT, cfg.VKITTI_DATASET.ORIGINAL_SIZE.WIDTH),
            #     mode='nearest'
            # )[0,0,...]

            # img_semantic = F.interpolate(
            #     img_semantic.unsqueeze(0).unsqueeze(0).float(),
            #     size=(cfg.VKITTI_DATASET.ORIGINAL_SIZE.HEIGHT, cfg.VKITTI_DATASET.ORIGINAL_SIZE.WIDTH),
            #     mode='nearest'
            # )[0,0,...]

            # Create segment_info data
            img_data['segments_info'] = []
            img_panoptic = img_panoptic.cpu().numpy()
            img_semantic = img_semantic.cpu().numpy()
            rgb_shape = tuple(list(img_panoptic.shape) + [3])
            rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
            
            for inst in np.unique(img_panoptic):
                if inst == 0:
                    continue
                rgb = randRGB(int(inst))
                locs = np.where(img_panoptic == inst)
                rgb_map[locs] = rgb
                img_data['segments_info'].append(
                    {
                        'id': int(inst),
                        'category_id': int(inst)
                                       if inst < 1000
                                       else int(inst / 1000)
                    }
                )
            # Save panotic_pred
            img_data['file_name'] = "{}_preds_panoptic.png".format(img_data['image_id'])
            # img = id2rgb(img_panoptic)
            img_to_save = Image.fromarray(rgb_map)
            img_to_save.save(os.path.join(pred_dir, img_data['file_name']))
            # Add annotation of a one image
            annotations.append(img_data)

            #Save semantic
            shape_semantic = img_semantic.shape[-2:]
            rgb_shape = tuple(list(shape_semantic) + [3])
            rgb_map_semantic = np.zeros(rgb_shape, dtype=np.uint8)
            for inst in np.unique(img_semantic):
                if inst == 0:
                    continue
                rgb = randRGB(int(inst))
                locs = np.where(img_semantic == inst)
                rgb_map_semantic[locs] = rgb

            filename_semantic = "{}_preds_panoptic_semantic.png".format(img_data['image_id'])
            img_to_save = Image.fromarray(rgb_map_semantic)
            img_to_save.save(os.path.join(pred_dir, filename_semantic))

            #Instance
            if instance != None:
                im = rgb_img.cpu().numpy().transpose((1, 2, 0))*0 #255 to overlay instances
                vis = Visualizer(im)
                instance = check_bbox_size(instance)
                if instance.has('pred_masks'):
                    masks = scale_resize_pad_masks(instance)
                    # p2d = (0, 0, 1, 1)
                    # masks = F.pad(masks, p2d, "constant", 0)
                    masks =np.asarray([mask.cpu().numpy() for mask in masks])
                    masks = np.where(masks > 0.5, int(1), 0)
                    instance.pred_masks = masks
                vis.draw_instance_predictions(instance)
                inst_output = vis.get_output()


                file_name = os.path.join(pred_dir, "{}_preds_panoptic_instance.png".format(img_data['image_id']))
                inst_output.save(file_name)
            

            #Save Depth
            if "depth" in output.keys():
                filename_depth = "{}_preds_panoptic_depth_gray.png".format(img_data['image_id'])
                depth = output["depth"][idx]
                save_image(depth, os.path.join(pred_dir, filename_depth), normalize=True)
                depth = depth.squeeze(0)
                shape = depth.shape
                out_depth_numpy = depth.cpu().numpy()/255
                filename_depth = "{}_preds_panoptic_depth".format(img_data['image_id'])
                save_fig(out_depth_numpy, pred_dir, filename_depth, shape)
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
    elif cfg.DATASET_TYPE == "forest":
        pred_path = os.path.join(cfg.FOREST_DATASET.DATASET_PATH.ROOT, cfg.FOREST_DATASET.DATASET_PATH.PRED_JSON)

    json_data={}
    json_data['annotations'] = annotations
    with open(pred_path, "w") as f:
        f.write(json.dumps(json_data))