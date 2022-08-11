import os
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)
from pytorch_lightning.utilities.model_summary  import ModelSummary
from detectron2.config import get_cfg
from detectron2.utils.events import _CURRENT_STORAGE_STACK, EventStorage


from efficientps import EffificientPS
from utils.add_custom_params import add_custom_params
from utils.dataloader_2_coco_panoptic import dataloader_2_coco_panoptic
from datasets.vkitti_dataset import get_dataloaders



def train(args):
    
    # Retrieve Config and and custom base parameter
    cfg = get_cfg()
    add_custom_params(cfg)
    cfg.merge_from_file(args.config)
    cfg.NUM_GPUS = args.ngpus
    
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

    print("Converting dataloader to coco_panoptic json")
    dataloader_2_coco_panoptic(cfg, valid_loader)
    # Create model or load a checkpoint
    if os.path.exists(cfg.CHECKPOINT_PATH_TRAINING):
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        print("Loading model from {}".format(cfg.CHECKPOINT_PATH_TRAINING))
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        efficientps = EffificientPS.load_from_checkpoint(cfg=cfg,
            checkpoint_path=cfg.CHECKPOINT_PATH_TRAINING)
    else:
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        print("Creating a new model")
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        efficientps = EffificientPS(cfg)
        cfg.CHECKPOINT_PATH_TRAINING = None

    # logger.info(efficientps.print)
    ModelSummary(efficientps, max_depth=-1)
    # Callbacks / Hooks
    early_stopping = EarlyStopping('PQ', patience=30, mode='max')
    checkpoint = ModelCheckpoint(monitor='PQ',
                                 mode='max',
                                 dirpath=cfg.CALLBACKS.CHECKPOINT_DIR,
                                 save_last=True,
                                 verbose=True,)

    # Create a pytorch lighting trainer
    trainer = pl.Trainer(
        # weights_summary='full',
        # auto_lr_find=True,
        log_every_n_steps=2500,
        gpus=args.ngpus,
        # distributed_backend='ddp',
        accelerator='ddp',
        num_sanity_val_steps=0,
        fast_dev_run=False if cfg.SOLVER.FAST_DEV_RUN is None else cfg.SOLVER.FAST_DEV_RUN,
        callbacks=[early_stopping, checkpoint],
        # precision=cfg.PRECISION,
        resume_from_checkpoint=cfg.CHECKPOINT_PATH_TRAINING,
        # gradient_clip_val=0,
        accumulate_grad_batches=cfg.SOLVER.ACCUMULATE_GRAD
    )
    logger.addHandler(logging.StreamHandler())
    # trainer.tune(efficientps, train_loader, valid_loader)
    trainer.fit(efficientps, train_loader, val_dataloaders=valid_loader)
