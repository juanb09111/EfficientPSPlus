import os
import logging
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
from detectron2.config import get_cfg
from detectron2.utils.events import _CURRENT_STORAGE_STACK, EventStorage


from efficientps import Instance
from utils.add_custom_params import add_custom_params
from datasets.vkitti_dataset import get_dataloaders
from datasets.vkitti_cats import obj_categories



def inference(args):
    
    # Retrieve Config and and custom base parameter
    cfg = get_cfg()
    add_custom_params(cfg)
    cfg.merge_from_file(args.config)
    cfg.NUM_GPUS = torch.cuda.device_count()

    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    logger = logging.getLogger("pytorch_lightning.core")
    if not os.path.exists(cfg.CALLBACKS.CHECKPOINT_DIR):
        os.makedirs(cfg.CALLBACKS.CHECKPOINT_DIR)
    logger.addHandler(logging.FileHandler(
        os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR,"core.log"), mode='w'))
    # with open(args.config) as file:
    #     logger.info(file.read())
    # Initialise Custom storage to avoid error when using detectron 2
    _CURRENT_STORAGE_STACK.append(EventStorage())

    #Get dataloaders
    if cfg.DATASET_TYPE == "vkitti2":
        train_loader, valid_loader, _ = get_dataloaders(cfg)
        categories = obj_categories

    # Create model or load a checkpoint
    if os.path.exists(cfg.CHECKPOINT_PATH_INFERENCE):
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        print("Loading model from {}".format(cfg.CHECKPOINT_PATH_INFERENCE))
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        efficientps_instance = Instance.load_from_checkpoint(cfg=cfg,
            checkpoint_path=cfg.CHECKPOINT_PATH_INFERENCE, categories=categories)
    else:
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        print("Creating a new model")
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        efficientps_instance = Instance(cfg, categories=categories)
        cfg.CHECKPOINT_PATH_INFERENCE = None

    ModelSummary(efficientps_instance, max_depth=-1) 
    trainer = pl.Trainer(
        log_every_n_steps=np.floor(len(valid_loader)/(cfg.BATCH_SIZE*torch.cuda.device_count())) -1,
        devices=list(range(torch.cuda.device_count())),
        strategy="ddp",
        accelerator='gpu',
        num_sanity_val_steps=0,
        # precision=cfg.PRECISION,
        resume_from_checkpoint=cfg.CHECKPOINT_PATH_INFERENCE,
        # gradient_clip_val=0,
        accumulate_grad_batches=cfg.SOLVER.ACCUMULATE_GRAD
    )
    logger.addHandler(logging.StreamHandler())

    
    efficientps_instance.eval()
    with torch.no_grad():
        predictions = trainer.predict(efficientps_instance, dataloaders=train_loader)

