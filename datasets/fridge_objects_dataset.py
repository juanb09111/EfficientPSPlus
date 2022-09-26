from pathlib import Path

import numpy as np
import torch
from torchvision.datasets.coco import CocoDetection
from detectron2.structures import Instances, BitMasks, Boxes
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

class EffDetCOCODataset(CocoDetection):
    def __init__(self, img_dir: Path, annotation_path: Path, transforms):
        super().__init__(root=str(img_dir), annFile=str(annotation_path))
        self.det_transforms = transforms

    def __getitem__(self, index):
        img, targets = super().__getitem__(index)
        img = np.array(img)
        
        img_ids = [t["image_id"] for t in targets] 
        #boxes
        bboxes = [target["bbox"] for target in targets]
        # x1,y1,w,h -> x1,y1,x2,y2
        bboxes = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in bboxes]

        # Labels must be 0 indexed!
        labels = [target["category_id"] - 1 for target in targets]
        transformed = self.det_transforms(image=img, bboxes=bboxes, labels=labels)
        transformed_img = transformed["image"]
        transformed_labels = transformed["labels"]

        _, new_h, new_w = transformed_img.shape

        instance = Instances((new_h, new_w))
        instance.gt_boxes = Boxes(torch.as_tensor(np.array(transformed["bboxes"]), dtype=torch.float32))
        instance.gt_classes = torch.as_tensor(torch.as_tensor(transformed_labels))

        return {
            "image": transformed_img,
            "image_id": torch.tensor([img_ids[0]]),
            "img_size": (new_h, new_w),
            'instance': instance
        }


def get_train_transforms(img_size: int) -> A.Compose:
    """get data transformations for train set
    Args:
        img_size (int): image size to resize input data
    Returns:
        A.Compose: whole data transformations to apply
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Resize(height=img_size, width=img_size),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_val_transforms(img_size: int):
    """get data transformations for val set
    Args:
        img_size (int): image size to resize input data
    Returns:
        A.Compose: whole data transformations to apply
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )




class EfficientDetDataModule(LightningDataModule):
    """LightningDataModule used for training EffDet
     This supports COCO dataset input
    Args:
        img_dir (Path): image directory
        annotation_dir (Path): annoation directory
        num_workers (int): number of workers to use for loading data
        batch_size (int): batch size
        img_size (int): image size to resize input data to during data
         augmentation
    """

    def __init__(
        self,
        img_dir: Path,
        annotation_dir: Path,
        num_workers: int,
        batch_size: int,
        img_size: int,
    ):
        super().__init__()
        self.train_transforms = get_train_transforms(img_size=img_size)
        self.val_transforms = get_val_transforms(img_size=img_size)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir

    def train_dataset(self) -> EffDetCOCODataset:
        return EffDetCOCODataset(
            img_dir=self.img_dir,
            annotation_path=self.annotation_dir,
            transforms=get_train_transforms(img_size=self.img_size),
        )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return train_loader

    def val_dataset(self) -> EffDetCOCODataset:
        return EffDetCOCODataset(
            img_dir=self.img_dir,
            annotation_path=self.annotation_dir,
            transforms=get_val_transforms(img_size=self.img_size),
        )

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.val_dataset()
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return val_loader
    
    def predict_dataset(self) -> EffDetCOCODataset:
        return EffDetCOCODataset(
            img_dir=self.img_dir,
            annotation_path=self.annotation_dir,
            transforms=get_val_transforms(img_size=self.img_size),
        )
    
    def predict_dataloader(self) -> DataLoader:
        pred_dataset = self.predict_dataset()
        pred_loader = torch.utils.data.DataLoader(
            pred_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return pred_loader

    @staticmethod
    def collate_fn(batch):      
        return {
            'image': torch.stack([i['image'] for i in batch]),
            'image_id': [i['image_id'] for i in batch],
            'instance': [i['instance'] for i in batch]
        }

# visualize
# import torchvision.transforms.functional as F
# from albumentations.pytorch.transforms import ToTensorV2
# from torchvision.utils import draw_bounding_boxes
# import os


# img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "odFridgeObjects", "images")
# annotation_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "odFridgeObjects", "ann_clean.json")
# id2label = {1: "can", 2: "carton", 3: "milk_bottle", 4: "water_bottle"}

# dataset = EffDetCOCODataset(
#     img_dir=img_dir,
#     annotation_path=annotation_dir,
#     transforms=get_val_transforms(img_size=512),
# )

# for i in range(128):
#     target = dataset[i]
#     img = target["image"]*255
#     instance = target["instance"]
#     labels = [id2label[id] for id in instance.get("gt_classes").tolist()]
#     boxes = instance.get("gt_boxes").tensor
#     # print(img, boxes, labels)
#     img = draw_bounding_boxes(
#         img.to(torch.uint8), boxes, labels, colors="Turquoise", width=2
#     )
#     img = F.to_pil_image(img.detach())

#     img.save("datasets/odFridgeObjects/test_val/{}.png".format(i),"PNG")