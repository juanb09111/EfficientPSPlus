import torch
import torch.nn.functional as F
from detectron2.structures import Instances
import os.path
import matplotlib.pyplot as plt


def dataset_mapping(cfg):

    if cfg.DATASET_TYPE == "vkitti2":
        stuff_classes = cfg.VKITTI_DATASET.STUFF_CLASSES
        instance_train_id_to_eval_id = [12, 13, 14]
        semantic_train_id_to_eval_id = [15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        # stuff_train_id_to_eval_id = [15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    elif cfg.DATASET_TYPE == "forest":
        stuff_classes = cfg.FOREST_DATASET.STUFF_CLASSES
        instance_train_id_to_eval_id = [2, 5, 6, 7, 8]
        semantic_train_id_to_eval_id = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        mlb_mapping = [2, 5, 6, 7, 8]
        # stuff_train_id_to_eval_id = 
       
    
    return {
        "stuff_classes": stuff_classes,
        "instance_train_id_to_eval_id": instance_train_id_to_eval_id,
        "semantic_train_id_to_eval_id": semantic_train_id_to_eval_id, 
        "mlb_mapping": mlb_mapping
    }

def panoptic_segmentation_module(cfg, outputs, device):
    """
    Take output of both semantic and instance head and combine them to create
    panoptic predictions.

    Note there is no need to check for threshold score, compute overlap and
    sorted scores. Since Detectron2 inference function already has the
    `SCORE_THRESH_TEST` and `NMS_THRESH_TEST` that does those action. Furthermore
    all prediction are sorted reated to their scores

    Args:
    - cfg (Config) : Config object
    - outputs (dict) : Inference output of our model
    - device : Device used by the lightning module

    Returns:
    - canvas (tensor) : [B, H, W] Panoptic predictions
    """
    # If no instance prediction pass the threshold score > 0.5 IoU > 0.5
    # Returns the argmax of semantic logits
    # outputs["instance"] = None
    # print("instance: ", outputs["instance"])
    # for i, sem in enumerate(outputs["semantic"]):
    #     plt.imshow(torch.argmax(sem, dim=0).cpu().numpy())
    #     plt.savefig(os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR, "semantic_output_epoch_{}_idx_{}.png".format(epoch, i)))

    dataset_map = dataset_mapping(cfg)
    if "instance" not in outputs.keys():
        return compute_output_only_semantic(cfg, outputs['semantic'])
    if outputs['instance'] is None: 
        return compute_output_only_semantic(cfg, outputs['semantic'])
    panoptic_result = []
    # Loop on Batch images / Instances
    for i, instance in enumerate(outputs['instance']):
        instance = check_bbox_size(instance)
        # If there is no proposal after the check compute panoptic output with
        # semantic information only
        if len(instance.pred_boxes.tensor) == 0:
            panoptic_result.append(
                        compute_output_only_semantic(cfg, outputs['semantic'][i]))
            continue
        semantic = outputs['semantic'][i]
        
        # Preprocessing
        Mla = scale_resize_pad(instance).to(device)
        # Compute instances
        if cfg.DATASET_TYPE == "vkitti2":
            Mlb = create_mlb(cfg, semantic, instance).to(device)
        elif cfg.DATASET_TYPE == "forest":
            Mlb = create_mlb_forest(cfg, semantic, instance).to(device)
        
        Fl = compute_fusion(Mla, Mlb)
        # First merge instances with stuff predictions
        semantic_stuff_logits = semantic[:dataset_map["stuff_classes"],:,:]
        # print("SHAPE", semantic.shape, Fl.shape, semantic_stuff_logits.shape)
        # print(torch.max(Fl), torch.max(semantic_stuff_logits), torch.min(Fl), torch.min(semantic_stuff_logits))
        # p2d = (0, 0, 1, 1)
        # Fl = F.pad(Fl, p2d, "constant", 0)
        inter_logits = torch.cat([semantic_stuff_logits, Fl], dim=0)
        inter_preds = torch.argmax(F.softmax(inter_logits, dim=0), dim=0)
        # Create canvas and merge everything
        canvas = create_canvas_thing(cfg, inter_preds, instance)

        if cfg.DATASET_TYPE == "vkitti2":
            canvas = add_stuff_from_semantic_vkitti(cfg, canvas, semantic)
        elif cfg.DATASET_TYPE == "forest":
            canvas = add_stuff_from_semantic_forest(cfg, canvas, semantic)
        
        
        panoptic_result.append(canvas)

    return torch.stack(panoptic_result)

def check_bbox_size(instance):
    """
    In some cases the width or height of a predicted bbox is 0. This function
    check all dimension and remove instances having this issue.
    Args:
    - instance (Instance) : detectron2 Instance object with prediction
    Returns:
    - new_instance (Instance) : dectron2 Instance with filtered prediction
    """
    new_instance = Instances(instance.image_size)
    boxes = instance.pred_boxes.tensor
    masks = instance.pred_masks
    inds = []
    for i, box in enumerate(boxes):
        # Retrieve bbox dimension
        box = torch.round(box).cpu().numpy()
        w = int(box[2]) - int(box[0])
        h = int(box[3]) - int(box[1])
        if h == 0 or w == 0:
            continue
        inds.append(i)
    new_instance.pred_masks = instance.pred_masks[inds]
    new_instance.pred_boxes = instance.pred_boxes[inds]
    new_instance.pred_classes = instance.pred_classes[inds]
    new_instance.scores = instance.scores[inds]
    return new_instance

def scale_resize_masks(instance):
    """
    In order to use both semantic and instances, mask must be rescale and fit
    the dimension of the bboxes predictions.
    Args:
    - instance (Instance) : an Instance object from detectron containg all
                            proposal bbox, masks, classes and scores
    """
    
    boxes = instance.pred_boxes.tensor
    masks = instance.pred_masks
    resized_masks = []
    # Loop on proposal
    for box, mask in zip(boxes, masks):
        # Retrieve bbox dimension
        box = torch.round(box).cpu().numpy()
        w = int(box[2]) - int(box[0])
        h = int(box[3]) - int(box[1])
        # Resize mask to bbox dimension
        # print("before", mask.shape)
        mask = F.interpolate(mask.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
        mask = mask[0,0,...]
        resized_masks.append(mask)
    return resized_masks

def scale_resize_pad_masks(instance):
    """
    In order to use both semantic and instances, mask must be rescale and fit
    the dimension of the bboxes predictions.
    Args:
    - instance (Instance) : an Instance object from detectron containg all
                            proposal bbox, masks, classes and scores
    """
    
    boxes = instance.pred_boxes.tensor
    masks = instance.pred_masks
    resized_masks = []
    # print("pred_classes", instance.pred_classes)
    # Loop on proposal
    if len(boxes) == 0:
        # print("NO OBJECTS DETECTED", instance.pred_classes)
        return torch.zeros((0), dtype=torch.bool)
    for box, mask in zip(boxes, masks):
        # Retrieve bbox dimension
        box = torch.round(box).cpu().numpy()
        w = int(box[2]) - int(box[0])
        h = int(box[3]) - int(box[1])
        # Resize mask to bbox dimension
        # print("before", mask.shape)
        mask = F.interpolate(mask.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
        mask = mask[0,0,...]

        canva = torch.zeros(instance.image_size)
        canva[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = mask
        resized_masks.append(canva)
    return torch.stack(resized_masks)

def scale_resize_pad(instance):
    """
    In order to use both semantic and instances, mask must be rescale and fit
    the dimension of the bboxes predictions.
    Args:
    - instance (Instance) : an Instance object from detectron containg all
                            proposal bbox, masks, classes and scores
    """
    Mla = []
    boxes = instance.pred_boxes.tensor
    masks = instance.pred_masks
    # Loop on proposal
    for box, mask in zip(boxes, masks):
        # Retrieve bbox dimension
        box = torch.round(box).cpu().numpy()
        w = int(box[2]) - int(box[0])
        h = int(box[3]) - int(box[1])
        # Resize mask to bbox dimension
        # print("before", mask.shape)
        mask = F.interpolate(mask.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
        mask = mask[0,0,...]
        # print("after", mask.shape)
        # Start from an empty canvas to have padding
        canva = torch.zeros(instance.image_size)
        # Fit the upsample mask in the bbox prediction position
        canva[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = mask
        Mla.append(canva)
    return torch.stack(Mla)

def create_mlb(cfg, semantic, instance):
    """
    Create the semantic logit corresponding to each class prediction.
    Args:
    - semantic (tensor) : Semantic logits of one image
    - instance (Instance) : Instance object with all instance prediction for one
                            image
    Returns:
    - Mlb (tensor) : dim[Nb of prediction, H, W]
    """
    Mlb = []
    boxes = instance.pred_boxes.tensor
    classes = instance.pred_classes
    dataset_map = dataset_mapping(cfg)
    # print("classes", classes)
    for bbox, cls in zip(boxes, classes):
        # Start from a black image
        canva = torch.zeros(instance.image_size)
        # Add the semantic value from the predicted class at the predicted bbox location
        canva[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = \
            semantic[cls+dataset_map["stuff_classes"],int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        Mlb.append(canva)
    return torch.stack(Mlb)

def create_mlb_forest(cfg, semantic, instance):
    """
    Create the semantic logit corresponding to each class prediction.
    Args:
    - semantic (tensor) : Semantic logits of one image
    - instance (Instance) : Instance object with all instance prediction for one
                            image
    Returns:
    - Mlb (tensor) : dim[Nb of prediction, H, W]
    """
    Mlb = []
    boxes = instance.pred_boxes.tensor
    classes = instance.pred_classes
    dataset_map = dataset_mapping(cfg)
    # print("classes", classes)
    for bbox, cls in zip(boxes, classes):
        # Start from a black image
        canva = torch.zeros(instance.image_size)
        # Add the semantic value from the predicted class at the predicted bbox location
        canva[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = \
            semantic[dataset_map["mlb_mapping"][cls],int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        Mlb.append(canva)
    return torch.stack(Mlb)


def compute_fusion(Mla, Mlb):
    """
    Compute the Hadamard product of the sum of sigmoid and the sum of Mla and
    Mlb. The Hadamard product is a fancy name for element-wise product.
    Args:
    - Mla (tensor) : Instance logit preprocess see `scale_resize_pad`
    - Mlb (tensor) : Semantic logits preprocess see `create_mlb`
    Returns:
    - Fl (tensor) : Fused mask logits
    """
    return (torch.sigmoid(Mla) + torch.sigmoid(Mlb)) * (Mla + Mlb)

def create_canvas_thing(cfg, inter_preds, instance):
    """
    From the intermediate prediction retrieve only the logits corresponding to
    thing classes.
    Args:
    -inter_preds (tensor): intermediate prediction
    -instance (Instance) : Instance object used to retrieve each classes of
                           instance prediction
    Returns:
    -canvas (tensor) : panoptic prediction containing only thing classes
    """
    dataset_map = dataset_mapping(cfg)
    
    # init to a number not in class category id
    canvas = torch.zeros_like(inter_preds)
    # Retrieve classes of all instance prediction (sorted by detectron2)
    classes = instance.pred_classes
    # instance_train_id_to_eval_id = [24, 25, 26, 27, 28, 31, 32, 33, 0]
    # instance_train_id_to_eval_id = [12, 13, 14]
    # Used to label each instance incrementally
    track_of_instance = {}
    # Loop on instance prediction
    for id_instance, cls in enumerate(classes):
        # The stuff channel are the 11 first channel so we add an offset
        id_instance += dataset_map["stuff_classes"]
        # Compute mask for each instance and verify that no prediction has been
        # made
        mask = torch.where((inter_preds == id_instance) & (canvas==0))
        # If the instance is present on interpreds add its panoptic label to
        # the canvas and increment the id of instance
        if len(mask) > 0:
            nb_instance = track_of_instance.get(int(cls), 0)
            canvas[mask] = dataset_map["instance_train_id_to_eval_id"][cls] * 1000 + nb_instance
            track_of_instance.update({int(cls):nb_instance+1})
    return canvas

def compute_output_only_semantic(cfg, semantic):
    """
    In case where no instance are suitable, we are returning the panoptic
    prediction base only on the semantic outputs.
    This is usefull mainly at the beginning of the training.
    Args:
    - semantic (tensor) : Output of the semantic head (either for the full
                          batch or for one image)
    """
    print("compute_output_only_semantic")
    dataset_map = dataset_mapping(cfg)
    
    # semantic_train_id_to_eval_id = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
    #                             23, 24, 25, 26, 27, 28, 31, 32, 33, 0]
    # semantic_train_id_to_eval_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # semantic_train_id_to_eval_id = [15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    if len(semantic.shape) == 3:
        semantic_output = torch.argmax(semantic, dim=0)
    else:
        semantic_output = torch.argmax(semantic, dim=1)
    # apply reversed to avoid issue with reindexing the value
    # print("output uniques: ", torch.unique(semantic_output))
    for train_id in reversed(torch.unique(semantic_output)):
        mask = torch.where(semantic_output == train_id)
        # Create panoptic ids for instance thing or stuff
        if train_id >= dataset_map["stuff_classes"]:  #12
            semantic_output[mask] = dataset_map["semantic_train_id_to_eval_id"][train_id] * 1000
        else:
            semantic_output[mask] = dataset_map["semantic_train_id_to_eval_id"][train_id]
    # print("1", semantic_output.shape)
    return semantic_output

def add_stuff_from_semantic_vkitti(cfg, canvas, semantic):
    """
    Compute the semantic output. If the output is not overlap with an existing
    prediction on the canvas (ie with a instance prediction) and the are is
    above the defined treshold, add the panoptic label of the stuff class
    on the canvas
    Args:
    - canvas (torch): canvas containing the thing class predictions
    - semantic (torch): logit output from the semantic head
    Return:
    - canvas (torch): Final panoptic prediction for an image
    """
    # Link between semantic and stuff classes in semantic prediction instance
    # classes have higher class training values
    # stuff_train_id_to_eval_id = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23]
    stuff_train_id_to_eval_id = [15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    semantic_output = torch.argmax(F.softmax(semantic, dim=0), dim=0)
    # Reverse to avoid overwrite classes information
    for train_id in reversed(torch.unique(semantic_output)):
        # If the detected section is a thing
        if train_id >= len(stuff_train_id_to_eval_id):
            continue
        # Compute mask where semantic is present and no things has been predicted
        mask = torch.where((semantic_output == train_id) & (canvas == 0))
        # Check the area is large enough
        if len(mask[0]) > cfg.INFERENCE.AREA_TRESH:
            # Compute mask where there is no thing classes
            canvas[mask] = stuff_train_id_to_eval_id[train_id]
    return canvas


def add_stuff_from_semantic_forest(cfg, canvas, semantic):
    """
    Compute the semantic output. If the output is not overlap with an existing
    prediction on the canvas (ie with a instance prediction) and the are is
    above the defined treshold, add the panoptic label of the stuff class
    on the canvas
    Args:
    - canvas (torch): canvas containing the thing class predictions
    - semantic (torch): logit output from the semantic head
    Return:
    - canvas (torch): Final panoptic prediction for an image
    """
    # Link between semantic and stuff classes in semantic prediction instance
    # classes have higher class training values
    instance_train_ids = [2, 5, 6, 7, 8]
    stuff_train_id_to_eval_id = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    semantic_output = torch.argmax(F.softmax(semantic, dim=0), dim=0)
    # Reverse to avoid overwrite classes information
    for train_id in reversed(torch.unique(semantic_output)):
        # If the detected section is a thing
        if train_id in instance_train_ids:
            continue
        # Compute mask where semantic is present and no things has been predicted
        mask = torch.where((semantic_output == train_id) & (canvas == 0))
        # Check the area is large enough
        if len(mask[0]) > cfg.INFERENCE.AREA_TRESH:
            # Compute mask where there is no thing classes
            canvas[mask] = stuff_train_id_to_eval_id[train_id]
    return canvas