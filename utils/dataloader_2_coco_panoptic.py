import json
import os.path
def dataloader_2_coco_panoptic(cfg, dataloader):

    gt_panoptic_json = os.path.join(cfg.VKITTI_DATASET.DATASET_PATH.ROOT, cfg.VKITTI_DATASET.DATASET_PATH.COCO_PANOPTIC_ANNOTATION)

    with open(gt_panoptic_json, 'r') as f:
        data = json.load(f)

    images_gt = data["images"]
    annotations_gt = data["annotations"]
    categories_gt = data["categories"]

    valid_json = os.path.join(cfg.VKITTI_DATASET.DATASET_PATH.ROOT, cfg.VKITTI_DATASET.DATASET_PATH.VALID_JSON)

    valid_data = {"categories":categories_gt, "images":[], "annotations":[]}
    images = []
    annotations = []
    for batch in dataloader:

        image_ids = batch["image_id"]
        for im_id in image_ids:
            image = list(filter(lambda im: im["id"] == im_id.item(), images_gt))[0]
            anns = list(filter(lambda ann: ann["image_id"] == im_id.item(), annotations_gt))
            
            images.append(image)
            annotations.extend(anns)
    
    valid_data["annotations"] = annotations
    valid_data["images"] = images

    with open(valid_json, 'w') as f:
        json.dump(valid_data, f)
