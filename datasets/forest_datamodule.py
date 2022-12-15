
import os
import os.path
from pathlib import Path
import math
import torch
from pytorch_lightning import LightningDataModule
from pycocotools.coco import COCO
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
from utils.get_forest_files import get_forest_files
from utils.show_ann import visualize_masks, visualize_bboxes
from datasets.forest_cats import mapping
from torchvision.utils import save_image
from torch.utils.data import random_split, DataLoader
from detectron2.structures import Instances, BitMasks, Boxes

import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

torch.manual_seed(0)

class ForestDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transforms, split="train"):
        
        self.cfg = cfg
        # Read config
        root = cfg.FOREST_DATASET.DATASET_PATH.ROOT
        self.split = split
        if split == "train":
            ann_path = os.path.join(root, cfg.FOREST_DATASET.DATASET_PATH.COCO_ANNOTATION_TRAIN)
            self.imgs_root = os.path.join(root, cfg.FOREST_DATASET.DATASET_PATH.RGB_TRAIN)

            self.depth_full_root = os.path.join(root, cfg.FOREST_DATASET.DATASET_PATH.DEPTH_TRAIN)
            self.depth_gt_root = os.path.join(root, cfg.FOREST_DATASET.DATASET_PATH.DEPTH_TRAIN)
            self.depth_proj_root = os.path.join(root, cfg.FOREST_DATASET.DATASET_PATH.DEPTH_TRAIN)
            self.semantic_root = os.path.join(root, cfg.FOREST_DATASET.DATASET_PATH.SEMANTIC_TRAIN)
        elif split == "val":
            ann_path = os.path.join(root, cfg.FOREST_DATASET.DATASET_PATH.COCO_ANNOTATION_VAL)
            self.imgs_root = os.path.join(root, cfg.FOREST_DATASET.DATASET_PATH.RGB_VAL)

            self.depth_full_root = os.path.join(root, cfg.FOREST_DATASET.DATASET_PATH.DEPTH_VAL)
            self.depth_gt_root = os.path.join(root, cfg.FOREST_DATASET.DATASET_PATH.DEPTH_VAL)
            self.depth_proj_root = os.path.join(root, cfg.FOREST_DATASET.DATASET_PATH.DEPTH_VAL)
            self.semantic_root = os.path.join(root, cfg.FOREST_DATASET.DATASET_PATH.SEMANTIC_VAL)
        
        self.coco = COCO(ann_path)

        self.semantic_imgs = get_forest_files(self.semantic_root, "png")
        self.depth_full_imgs = get_forest_files(self.depth_full_root, "png")
        self.depth_gt_imgs = get_forest_files(self.depth_gt_root, "png")
        self.depth_proj_imgs = get_forest_files(self.depth_proj_root, "png")

        # Filter ids by scenes
        im_ids = list(sorted(self.coco.imgs.keys()))
        rgb_images = self.coco.loadImgs(ids=im_ids)

        self.ids = list(map(lambda im_obj: im_obj["id"], rgb_images))

        catIds = self.coco.getCatIds()
        categories = self.coco.loadCats(catIds)
        self.categories = list(map(lambda x: x['name'], categories))
        self.bg_categories_ids = self.coco.getCatIds(supNms="background")
        bg_categories = self.coco.loadCats(self.bg_categories_ids)
        self.bg_categories = list(map(lambda x: x['name'], bg_categories))

        self.obj_categories_ids = self.coco.getCatIds(supNms="object")
        obj_categories = self.coco.loadCats(self.obj_categories_ids)
        self.obj_categories = list(map(lambda x: x['name'], obj_categories))

        print("Thing classes: ", self.obj_categories)
        print("Stuff classes: ", self.bg_categories)

        # self.depth_imgs = get_forest_files(self.depth_root, exclude, "png")

        self.transforms = transforms

        if cfg.FOREST_DATASET.MAX_SAMPLES != None:
            self.ids = self.ids[:cfg.FOREST_DATASET.MAX_SAMPLES]



    def find_k_nearest(self, lidar_fov):
        k_number = self.cfg.FOREST_DATASET.DEPTH.K
        b_lidar_fov = torch.unsqueeze(lidar_fov, dim=0)

        distances = torch.cdist(b_lidar_fov, b_lidar_fov, p=2)
        _, indices = torch.topk(distances, k_number + 1, dim=2, largest=False)
        indices = indices[:, :, 1:]  # B x N x 3

        return indices.squeeze_(0).long()

    def mask_to_class(self,mask):
        target = torch.from_numpy(mask)
        h,w = target.shape[0],target.shape[1]
        masks = torch.empty(h, w, dtype=torch.long)
        target = target.permute(2, 0, 1).contiguous()
        for k in mapping:
            idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3) 
            masks[validx] = torch.tensor(mapping[k], dtype=torch.long)
        I8 = (((masks.cpu().numpy()))).astype(np.uint8)
        im = Image.fromarray(I8)
        return im

    def get_coco_ann(self, index):

        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        
        obj_ann_ids = coco.getAnnIds(
            imgIds=img_id, catIds=self.obj_categories_ids)
            
        coco_annotation = coco.loadAnns(obj_ann_ids)

        # path for input image
        img_filename = coco.loadImgs(img_id)[0]['file_name']

        basename = img_filename.split(".")[-2]

        semantic_img_filename = [s for s in self.semantic_imgs if basename in s][0]        
        
        # depth_full_img_filename = [s for s in self.depth_full_imgs if basename in s][0]

        depth_gt_img_filename = [s for s in self.depth_gt_imgs if basename in s][0]
        
        depth_proj_img_filename = [s for s in self.depth_proj_imgs if basename in s][0]
        # print(semantic_img_filename)
        semantic_mask = Image.open(semantic_img_filename)
        semantic_mask = self.mask_to_class(np.array(semantic_mask))
        semantic_mask = np.asarray(semantic_mask, dtype=np.long)    

        # depth_full = np.asarray(Image.open(depth_full_img_filename))/255
        # depth_gt = np.asarray(Image.open(depth_gt_img_filename))
        # print("here2!!", np.unique(depth_gt))
        depth_gt = np.asarray(Image.open(depth_gt_img_filename))*50.0/255
        # # print("gt",np.unique(depth_gt), np.count_nonzero(depth_gt))

        
        depth_proj = np.asarray(Image.open(depth_proj_img_filename))*50.0/255
        # # print("proj",np.unique(depth_proj), np.count_nonzero(depth_proj))
        
        
        num_objs = len(coco_annotation)
        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        masks = []
        category_ids = []
        for i in range(num_objs):

            mask = coco.annToMask(coco_annotation[i])
            masks.append(mask)

            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

            category_id = coco_annotation[i]['category_id']
            label = coco.cats[category_id]['name']
            # print(label)
            #TODO: check whether or not to add background 0 label
            # print(self.obj_categories)
            # labels must be 0 indexed!
            labels.append(self.obj_categories.index(label))
            # print(labels)
            area = coco_annotation[i]['area']
            areas.append(area)

            iscrowd.append(coco_annotation[i]['iscrowd'])

            category_ids.append(category_id)

        # Tensorise img_id
        img_id = torch.tensor([img_id])

        category_ids = torch.as_tensor(category_ids, dtype=torch.int64)

        # Num of instance objects
        num_objs = torch.as_tensor(num_objs, dtype=torch.int64)
        # Annotation is in dictionary format
        ann = {}
        ann["boxes"] = boxes
        ann["labels"] = labels
        ann["image_id"] = img_id
        ann["area"] = areas
        ann["iscrowd"] = iscrowd
        ann["category_ids"] = category_ids
        ann["num_instances"] = num_objs
        ann['masks'] = masks

        ann["semantic_mask"] = semantic_mask
        # ann["depth_full"] = depth_full
        ann["depth_gt"] = depth_gt
        ann["depth_proj"] = depth_proj
        
        return img_filename, ann

    def __getitem__(self, index):

        img_filename, ann = self.get_coco_ann(index)

        basename = img_filename.split(".")[-2]

        if self.split == "train":
            img_filename = os.path.join(self.cfg.FOREST_DATASET.DATASET_PATH.ROOT, self.cfg.FOREST_DATASET.DATASET_PATH.RGB_TRAIN, img_filename)
        elif self.split == "val": 
            img_filename = os.path.join(self.cfg.FOREST_DATASET.DATASET_PATH.ROOT, self.cfg.FOREST_DATASET.DATASET_PATH.RGB_VAL, img_filename)
        source_img = np.asarray(Image.open(img_filename))
        image_id = ann["image_id"]
        
        if self.transforms is not None:
            
            transformed = self.transforms(
                image=source_img,
                masks=[*ann["masks"], ann['depth_gt'], ann['depth_proj'], ann['semantic_mask']],
                # masks=[*ann["masks"], ann["depth_full"], ann['depth_gt'], ann['depth_proj'], ann['semantic_mask']],
                bboxes=ann["boxes"],
                labels=ann["labels"]
            )
            
        
            source_img = transformed["image"]
            # depth_full = transformed["masks"][-4]
            # depth_full = torch.where(depth_full >= self.cfg.FOREST_DATASET.DEPTH.MAX_DEPTH, torch.tensor([
            #     0], dtype=torch.float64), depth_full)
            depth_gt = transformed["masks"][-3]
            depth_proj = transformed["masks"][-2]
            
            semantic_mask = np.asarray(transformed['masks'][-1].cpu().numpy(), dtype=np.long)

            num_boxes = len(transformed['bboxes'])
            instance = Instances(semantic_mask.shape)
            
            if num_boxes:
                instance_masks =[mask.cpu().numpy() for mask in transformed['masks'][:num_boxes]]
                instance.gt_masks = BitMasks(np.asarray(instance_masks))
                instance.gt_classes = torch.as_tensor(transformed['labels'])
                instance.gt_boxes = Boxes(transformed['bboxes'])
            else:
                instance.gt_masks = BitMasks(torch.Tensor([]).view(0,1,1))
                instance.gt_classes = torch.as_tensor([])
                instance.gt_boxes = Boxes([])

        
        imPts = torch.nonzero(depth_proj)
        inds = torch.randperm(imPts.shape[0])
        imPts = imPts[inds][:self.cfg.FOREST_DATASET.DEPTH.MAX_DEPTH_POINTS]

        # if imPts.shape[0] < self.max_points:
        #     self.max_points = imPts.shape[0]
        #     print(self.max_points)
            # sys.stdout.write("\rmax_points {}".format(self.max_points))

        virtual_lidar = torch.zeros((imPts.shape[0], 3))
        virtual_lidar[:, 0:2] = imPts
        virtual_lidar[:, 2] = depth_proj[imPts[:, 0], imPts[:, 1]]
        # print("HERE!!", torch.unique(virtual_lidar[:, 2]))
        mask = torch.zeros(depth_proj.shape[-2:], dtype=torch.bool)
        mask[imPts[:, 0], imPts[:, 1]] = True

        k_nn_indices = self.find_k_nearest(virtual_lidar)

        # # Remove points from depth_proj to the total number of points = MAX_DEPTH_POINTS
        depth_proj_ = torch.zeros_like(depth_proj)
        depth_proj_[imPts[:, 0], imPts[:, 1]] = depth_proj[imPts[:, 0], imPts[:, 1]]
        

        return {
            "image": source_img, 
            "semantic": semantic_mask,
            'instance': instance, 
            "image_id": image_id,
            "basename": basename,
            "n_instances": num_boxes,
            "mask": mask,
            "virtual_lidar": virtual_lidar,
            "sparse_depth": depth_proj_.unsqueeze_(0).float(), 
            "sparse_depth_gt": depth_gt.unsqueeze_(0).float(), 
            # "depth_full": depth_full.unsqueeze_(0).float(), 
            "k_nn_indices": k_nn_indices
        }
        

    def __len__(self):
        return len(self.ids)
    

def get_train_transforms(cfg):

    custom_transforms = []
    h_crop, w_crop = (cfg.FOREST_DATASET.RANDOMCROP.HEIGHT, cfg.FOREST_DATASET.RANDOMCROP.WIDTH)
    h_ccrop, w_ccrop = (cfg.FOREST_DATASET.CENTER_CROP.HEIGHT, cfg.FOREST_DATASET.CENTER_CROP.WIDTH)
    h_resize, w_resize = (cfg.FOREST_DATASET.RESIZE.HEIGHT, cfg.FOREST_DATASET.RESIZE.WIDTH)


    custom_transforms.append(A.Resize(height=h_resize, width=w_resize))
    # custom_transforms.append(A.RandomCrop(width=w_crop, height=h_crop))
    custom_transforms.append(A.CenterCrop(width=w_ccrop, height=h_ccrop))
    custom_transforms.append(A.HorizontalFlip(p=cfg.FOREST_DATASET.HFLIP))
    custom_transforms.append(A.Normalize(mean=cfg.FOREST_DATASET.NORMALIZE.MEAN, std=cfg.FOREST_DATASET.NORMALIZE.STD))
    custom_transforms.append(ToTensorV2())

    return A.Compose(custom_transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_val_transforms(cfg):

    custom_transforms = []
    h_crop, w_crop = (cfg.FOREST_DATASET.RANDOMCROP.HEIGHT, cfg.FOREST_DATASET.RANDOMCROP.WIDTH)
    h_ccrop, w_ccrop = (cfg.FOREST_DATASET.CENTER_CROP.HEIGHT, cfg.FOREST_DATASET.CENTER_CROP.WIDTH)
    h_resize, w_resize = (cfg.FOREST_DATASET.RESIZE.HEIGHT, cfg.FOREST_DATASET.RESIZE.WIDTH)


    custom_transforms.append(A.Resize(height=h_resize, width=w_resize))
    custom_transforms.append(A.CenterCrop(width=w_ccrop, height=h_ccrop))
    custom_transforms.append(A.Normalize(mean=cfg.FOREST_DATASET.NORMALIZE.MEAN, std=cfg.FOREST_DATASET.NORMALIZE.STD))
    custom_transforms.append(ToTensorV2())

    return A.Compose(custom_transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))




def test_dataset(cfg, dataset, item, split="train"):
    sample = dataset.__getitem__(item)
    basename = sample["basename"]
    source_img = sample["image"]
    instance = sample["instance"]
    semantic = sample["semantic"]

    mask = sample["mask"]

    sparse_depth = sample["sparse_depth"]
    sparse_depth_gt = sample["sparse_depth_gt"]

    save_image(source_img, os.path.join("forest_dataset_test/{}".format(split), "source_img_{}.png".format(basename)))
    save_image(sparse_depth, os.path.join("forest_dataset_test/{}".format(split), "sparse_depth_{}.png".format(basename)), normalize=True)
    save_image(sparse_depth_gt, os.path.join("forest_dataset_test/{}".format(split), "sparse_depth_gt_{}.png".format(basename)), normalize=True)
    save_image(mask.float(), os.path.join("forest_dataset_test/{}".format(split), "depth_mask_{}.png".format(basename)), normalize=True)
    


    # Visualize annotations
    image = source_img.cpu().numpy().transpose((1, 2, 0))*255
    image = visualize_masks(instance.get("gt_masks").tensor)
    image = visualize_bboxes(image, instance.get("gt_boxes"))
    cv2.imwrite(os.path.join("forest_dataset_test/{}".format(split), "masks_{}.png".format(basename)), image)

    cv2.imwrite(os.path.join("forest_dataset_test/{}".format(split), "semantic_gray_{}.png".format(basename)), semantic)
    plt.imshow(semantic)
    plt.savefig(os.path.join("forest_dataset_test/{}".format(split), "semantic_{}.png".format(basename)))
    


class ForestDataModule(LightningDataModule):
    """LightningDataModule used for training EffDet
     This supports COCO dataset input
    Args:
        cgf: config
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.BATCH_SIZE

        # for split in ["train", "val"]:
        #     # Test dataset on one sample
        #     train_set = ForestDataset(self.cfg, get_train_transforms(self.cfg), split=split)
        #     # print(len(train_set))
        #     for i in range(len(train_set)):
        #         test_dataset(cfg, train_set, i, split=split)
        #     print("Dataset test finished")

    def train_dataset(self) -> ForestDataset:

        return ForestDataset(self.cfg, get_train_transforms(self.cfg), split="train")

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        print("Number of training samples: {}".format(len(train_dataset)))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.cfg.FOREST_DATASET.SHUFFLE,
            pin_memory=True,
            drop_last=True,
            num_workers=4,
            collate_fn=self.collate_fn_wrapper(train_dataset),
        )

        return train_loader

    
    def val_dataset(self) -> ForestDataset:

        return ForestDataset(self.cfg, get_val_transforms(self.cfg), split="val")

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


    def predict_dataset(self) -> ForestDataset:

        return ForestDataset(self.cfg, get_val_transforms(self.cfg), split="val")

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
                'semantic': torch.as_tensor(np.asarray([i['semantic'] for i in batch])),
                'instance': [i['instance'] for i in batch],
                'image_id': [i['image_id'] for i in batch],
                'basename': [i['basename'] for i in batch],
                'mask': torch.stack([i['mask'] for i in batch]),
                'virtual_lidar': torch.stack([i['virtual_lidar'] for i in batch]),
                'sparse_depth': torch.stack([i['sparse_depth'] for i in batch]),
                # 'depth_full': torch.stack([i['depth_full'] for i in batch]),
                'k_nn_indices': torch.stack([i['k_nn_indices'] for i in batch]),
                'sparse_depth_gt': torch.stack([i['sparse_depth_gt'] for i in batch])
            }
        return collate_fn