
import os
import os.path
from pathlib import Path
import math
import torch
from pytorch_lightning import LightningDataModule
from ytvostools.ytvos import YTVOS
from ytvostools import mask
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random

from detectron2.structures import Instances, BitMasks, Boxes

import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

torch.manual_seed(0)
class youtubeDataset(torch.utils.data.Dataset):

    def __init__(self,
                 imgs_root,
                 annotation,
                 valid_annotation=None,
                 transforms=None,
                 n_samples=None):

        self.transforms = transforms
        self.imgs_root = imgs_root
        self.ytvos = YTVOS(annotation)


        self.vidIds = self.ytvos.getVidIds()
        self.videos = self.ytvos.loadVids(ids=self.vidIds)

        if valid_annotation != None:
            self.ytvos_valid = YTVOS(valid_annotation)
            self.vidIds_valid = self.ytvos_valid.getVidIds()
            self.videos_valid = self.ytvos_valid.loadVids(ids=self.vidIds_valid)

            self.videos = list(filter(lambda video: video not in self.videos_valid, self.videos))


        self.frames = []
        for video in self.videos:
            file_names = video["file_names"]
            video_id = video["id"]
            annIds = self.ytvos.getAnnIds(vidIds=[video_id])
            anns = self.ytvos.loadAnns(ids=annIds)

            num_objs = len(anns)
            video_cats = [anns[i]["category_id"] for i in range(num_objs)]
            video_iscrowd = [anns[i]["iscrowd"] for i in range(num_objs)]
           
            for idx, file_name in enumerate(file_names):
                frame_anns = [
                    el for el in
                    [anns[i]["segmentations"][idx] for i in range(num_objs)]
                    if el is not None
                ]
                frame_boxes = [
                    el for el in
                    [anns[i]["bboxes"][idx] for i in range(num_objs)]
                    if el is not None
                ]
                frame_areas = [
                    el
                    for el in [anns[i]["areas"][idx] for i in range(num_objs)]
                    if el is not None
                ]

                if len(frame_anns) > 0:
                    frame_cats = [
                        video_cats[j] for j in range(len(frame_anns))
                    ]
                    frame_iscrowd = [
                        video_iscrowd[j] for j in range(len(frame_anns))
                    ]

                    self.frames.append({
                        "file_name": file_name,
                        "anns": frame_anns,
                        "frame_cats": frame_cats,
                        "bboxes": frame_boxes,
                        "areas": frame_areas,
                        "iscrowd": frame_iscrowd,
                        "video_id": video_id
                    })

        if n_samples != None:
            self.frames = self.frames[:n_samples]

    def __getitem__(self, index):

        frame = self.frames[index]
        boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]]
                 for box in frame["bboxes"]]
        labels = frame["frame_cats"]
        areas = frame["areas"]
        iscrowd = frame["iscrowd"]
        video_id = frame["video_id"]
        masks = [
            mask.decode(mask.frPyObjects(rle, rle['size'][0], rle['size'][1]))
            for rle in frame["anns"]
        ]

        img_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", self.imgs_root, frame["file_name"])
        basename = img_filename.split(".")[-2]
        source_img = np.asarray(Image.open(img_filename))

        if self.transforms is not None:
            
            transformed = self.transforms(
                image=source_img,
                masks=boxes,
                bboxes=boxes,
                labels=labels
            )
        
        source_img = transformed["image"]
        num_boxes = len(transformed['bboxes'])
        instance = Instances(source_img.shape[-2:])

        if num_boxes:
                instance_masks =[mask.cpu().numpy() for mask in transformed['masks'][:num_boxes]]
                instance.gt_masks = BitMasks(np.asarray(instance_masks))
                instance.gt_classes = torch.as_tensor(transformed['labels'])
                instance.gt_boxes = Boxes(transformed['bboxes'])
        else:
            instance.gt_masks = BitMasks(torch.Tensor([]).view(0,1,1))
            instance.gt_classes = torch.as_tensor([])
            instance.gt_boxes = Boxes([])

        return {
            "image": source_img, 
            'instance': instance, 
            "basename": basename,
            "video_id": video_id, 
            "n_instances": num_boxes
        }
    
    def __len__(self):
        return len(self.frames)

def get_train_transforms(cfg):

    custom_transforms = []
    custom_transforms.append(A.Normalize(mean=cfg.YT_DATASET.NORMALIZE.MEAN, std=cfg.YT_DATASET.NORMALIZE.STD))
    custom_transforms.append(ToTensorV2())

    return A.Compose(custom_transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_val_transforms(cfg):
    custom_transforms = []

    custom_transforms.append(A.Normalize(mean=cfg.YT_DATASET.NORMALIZE.MEAN, std=cfg.YT_DATASET.NORMALIZE.STD))
    custom_transforms.append(ToTensorV2())

    return A.Compose(custom_transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))



def test_dataset(cfg, dataset, item):
    print("Dataset test")
    sample = dataset.__getitem__(item)
    basename = sample["basename"]
    video_id = sample["video_id"]
    source_img = sample["image"]

    instance = sample["instance"]

    save_image(source_img, os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR, "dataset_test_v2", "source_img_{}.png".format(basename)))
    

    # Visualize annotations
    image = source_img.cpu().numpy().transpose((1, 2, 0))*255
    image = visualize_masks(instance.get("gt_masks").tensor)
    image = visualize_bboxes(image, instance.get("gt_boxes"))
    cv2.imwrite(os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR, "dataset_test_v2", "masks_{}.png".format(basename)), image)
    print("Dataset test finished")


class YoutubeDataModule(LightningDataModule):
    """LightningDataModule used for training EffDet
     This supports COCO dataset input
    Args:
        cgf: config
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.BATCH_SIZE
        dataset_root = cgf.YT_DATASET.DATASET_PATH.ROOT

        imgs_root = os.path.join(root, cgf.YT_DATASET.DATASET_PATH.RGB_TRAIN)
        annotation = os.path.join(root, cgf.YT_DATASET.DATASET_PATH.ANN_TRAIN)
        
        train_set = youtubeDataset(imgs_root, annotation, transforms=get_train_transforms(self.cfg))
        test_dataset(cfg, train_set, 0)

    def train_dataset(self) -> youtubeDataset:

        return youtubeDataset(imgs_root, annotation, transforms=get_train_transforms(self.cfg))

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        print("Number of training samples: {}".format(len(train_dataset)))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.cfg.YT_DATASET.SHUFFLE,
            pin_memory=True,
            drop_last=True,
            num_workers=4,
            collate_fn=self.collate_fn_wrapper(train_dataset),
        )

        return train_loader

    
    def val_dataset(self) -> youtubeDataset:

        return youtubeDataset(imgs_root, annotation, transforms=get_val_transforms(self.cfg))

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.val_dataset()
        print("Number of eval samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=4,
            collate_fn=self.collate_fn_wrapper(val_dataset),
        )

        return val_loader


    def predict_dataset(self) -> youtubeDataset:

        return youtubeDataset(imgs_root, annotation, transforms=get_val_transforms(self.cfg))

    def predict_dataloader(self) -> DataLoader:
        predict_dataset = self.predict_dataset()
        print("Number of test samples: {}".format(len(predict_dataset)))
        predict_loader = torch.utils.data.DataLoader(
            predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=4,
            collate_fn=self.collate_fn_wrapper(predict_dataset),
        )

        return predict_loader

    @staticmethod
    def collate_fn_wrapper(dataset):
        def collate_fn(batch):
            len_batch = len(batch) # original batch length
            batch = list(filter (lambda x:x["n_instances"] > 0, batch)) # filter out all the Nones
            while len_batch > len(batch): # source all the required samples from the original dataset at random
                diff = len_batch - len(batch)
                for i in range(diff):
                    new_sample = dataset[np.random.randint(0, len(dataset))]
                    batch.append(new_sample)
                batch = list(filter (lambda x:x["n_instances"] > 0, batch))
            return {
                'image': torch.stack([i['image'] for i in batch]),
                'instance': [i['instance'] for i in batch],
                'video_id': [i['video_id'] for i in batch],
                'basename': [i['basename'] for i in batch]
            }
        return collate_fn