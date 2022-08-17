from detectron2.config import CfgNode

def add_custom_params(cfg):
    """
    In order to add custom config parameter in the .yaml those parameter must
    be initialised
    """
    # Model
    cfg.MODEL_CUSTOM = CfgNode()
    cfg.MODEL_CUSTOM.BACKBONE = CfgNode()
    cfg.MODEL_CUSTOM.BACKBONE.EFFICIENTNET_ID = 5
    cfg.MODEL_CUSTOM.BACKBONE.LOAD_PRETRAIN = False
    # DATASET
    cfg.NUM_CLASS = 15
    # cfg.DATASET_PATH = "/home/ubuntu/Elix/cityscapes"
    # cfg.TRAIN_JSON = "gtFine/cityscapes_panoptic_train.json"
    # cfg.VALID_JSON = "gtFine/cityscapes_panoptic_val.json"
    # cfg.PRED_DIR = "preds"
    # cfg.PRED_JSON = "cityscapes_panoptic_preds.json"
    #VKITTI DATASET
    cfg.VKITTI_DATASET = CfgNode()

    cfg.VKITTI_DATASET.HFLIP = 0.5

    cfg.VKITTI_DATASET.RANDOMCROP = CfgNode()
    cfg.VKITTI_DATASET.RANDOMCROP.HEIGHT = 512
    cfg.VKITTI_DATASET.RANDOMCROP.WIDTH = 1024

    cfg.VKITTI_DATASET.RESIZE = CfgNode()
    cfg.VKITTI_DATASET.RESIZE.HEIGHT = 512
    cfg.VKITTI_DATASET.RESIZE.WIDTH = 1024
    
    cfg.VKITTI_DATASET.ORIGINAL_SIZE = CfgNode()
    cfg.VKITTI_DATASET.ORIGINAL_SIZE.HEIGHT = 375
    cfg.VKITTI_DATASET.ORIGINAL_SIZE.WIDTH = 1242

    cfg.VKITTI_DATASET.CENTER_CROP = CfgNode()
    cfg.VKITTI_DATASET.CENTER_CROP.HEIGHT = 200
    cfg.VKITTI_DATASET.CENTER_CROP.WIDTH = 1000

    cfg.VKITTI_DATASET.NORMALIZE = CfgNode()
    cfg.VKITTI_DATASET.NORMALIZE.MEAN = (0.485, 0.456, 0.406)
    cfg.VKITTI_DATASET.NORMALIZE.STD = (0.229, 0.224, 0.225)

    cfg.VKITTI_DATASET.DEPTH = CfgNode()
    cfg.VKITTI_DATASET.DEPTH.K = 3
    cfg.VKITTI_DATASET.DEPTH.MAX_DEPTH = 50
    cfg.VKITTI_DATASET.DEPTH.SPARSITY = 0.05

    cfg.VKITTI_DATASET.DATASET_PATH = CfgNode()
    cfg.VKITTI_DATASET.STUFF_CLASSES = 12
    cfg.VKITTI_DATASET.SHUFFLE = True
    cfg.VKITTI_DATASET.MAX_SAMPLES = 100
    cfg.VKITTI_DATASET.SPLIT_DATASET= True
    cfg.VKITTI_DATASET.SPLITS= [0.80, 0.15, 0.05]
    cfg.VKITTI_DATASET.DATASET_PATH.ROOT = "datasets/vkitti2"
    cfg.VKITTI_DATASET.DATASET_PATH.RGB = "vkitti_2.0.3_rgb"
    cfg.VKITTI_DATASET.DATASET_PATH.SEMANTIC = "vkitti_2.0.3_classSegmentation"
    cfg.VKITTI_DATASET.DATASET_PATH.INSTANCE = "vkitti_2.0.3_instanceSegmentation"
    cfg.VKITTI_DATASET.DATASET_PATH.DEPTH = "vkitti_2.0.3_depth"
    cfg.VKITTI_DATASET.DATASET_PATH.COCO_ANNOTATION = "vkitti_2.0.3_coco.json"
    cfg.VKITTI_DATASET.DATASET_PATH.COCO_PANOPTIC_SEGMENTATION = "kitti_coco_panoptic"
    cfg.VKITTI_DATASET.DATASET_PATH.TWO_CH_PANOPTIC_SEGMENTATION = "kitti_2ch_panoptic"
    cfg.VKITTI_DATASET.DATASET_PATH.TWO_CH_IMAGE_JSON = "kitti_2ch_panoptic.json"
    cfg.VKITTI_DATASET.DATASET_PATH.COCO_PANOPTIC_ANNOTATION = "kitti_coco_panoptic.json"
    cfg.VKITTI_DATASET.DATASET_PATH.VALID_JSON = "kitti_coco.json"
    cfg.VKITTI_DATASET.DATASET_PATH.VALID_PRED_DIR = "preds_valid"
    cfg.VKITTI_DATASET.DATASET_PATH.PRED_DIR = "preds"
    cfg.VKITTI_DATASET.DATASET_PATH.PRED_JSON = "vkitti2_panoptic_predictions.json"
    cfg.VKITTI_DATASET.DATASET_PATH.PRED_DIR_SEMANTIC = "preds_semantic"
    cfg.VKITTI_DATASET.DATASET_PATH.PRED_DIR_INSTANCE = "preds_instance"
    cfg.VKITTI_DATASET.DATASET_PATH.PRED_JSON_INSTANCE = "vkitti2_instance_predictions.json"
    cfg.VKITTI_DATASET.DATASET_PATH.PRED_JSON_SEMANTIC = "vkitti2_semantic_predictions.json"
    
    cfg.VKITTI_DATASET.EXCLUDE = ["15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right"]

    #Dataset type
    cfg.DATASET_TYPE = "vkitti2"
    cfg.NUM_GPUS = 4
    # Transfom
    cfg.TRANSFORM = CfgNode()
    cfg.TRANSFORM.NORMALIZE = CfgNode()
    cfg.TRANSFORM.NORMALIZE.MEAN = (106.433, 116.617, 119.559)
    cfg.TRANSFORM.NORMALIZE.STD = (65.496, 67.6, 74.123)
    cfg.TRANSFORM.RESIZE = CfgNode()
    cfg.TRANSFORM.RESIZE.HEIGHT = 512
    cfg.TRANSFORM.RESIZE.WIDTH = 1024
    cfg.TRANSFORM.RANDOMCROP = CfgNode()
    cfg.TRANSFORM.RANDOMCROP.HEIGHT = 512
    cfg.TRANSFORM.RANDOMCROP.WIDTH = 1024
    cfg.TRANSFORM.HFLIP = CfgNode()
    cfg.TRANSFORM.HFLIP.PROB = 0.5
    # Solver
    cfg.SOLVER.NAME = "SGD"
    cfg.SOLVER.ACCUMULATE_GRAD = 1
    cfg.SOLVER.FAST_DEV_RUN = None
    cfg.SOLVER.BASE_LR_SEMANTIC = 0.0013182567385564075
    cfg.SOLVER.BASE_LR_INSTANCE = 0.00014454397707459272
    # Runner
    cfg.BATCH_SIZE = 2
    cfg.CHECKPOINT_PATH_TRAINING = ""
    cfg.CHECKPOINT_PATH_INFERENCE = ""
    cfg.PRECISION = 32
    # Callbacks
    cfg.CALLBACKS = CfgNode()
    cfg.CALLBACKS.CHECKPOINT_DIR = None
    # Inference
    cfg.INFERENCE = CfgNode()
    cfg.INFERENCE.AREA_TRESH = 0
