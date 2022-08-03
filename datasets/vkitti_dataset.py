# %%

import os.path
import math
import torch
from pycocotools.coco import COCO
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
from utils.get_vkitti_files import get_vkitti_files
from utils.show_ann import visualize_masks, visualize_bboxes, visualize_titles
from datasets.vkitti_cats import rgb_2_class, mapping
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.utils.data import random_split
from detectron2.structures import Instances, BitMasks, Boxes
import cv2
import time
# %%
torch.manual_seed(0)

class vkittiDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transforms):
        
        self.cfg = cfg

        
        # Read config
        exclude = cfg.VKITTI_DATASET.EXCLUDE
        root = cfg.VKITTI_DATASET.DATASET_PATH.ROOT
        self.imgs_root = os.path.join(root, cfg.VKITTI_DATASET.DATASET_PATH.RGB)
        # self.depth_root = os.path.join(root, cfg.VKITTI_DATASET.DATASET_PATH.DEPTH)
        self.semantic_root = os.path.join(root, cfg.VKITTI_DATASET.DATASET_PATH.SEMANTIC)
        self.coco = COCO(os.path.join(root, cfg.VKITTI_DATASET.DATASET_PATH.COCO_ANNOTATION))

        self.semantic_imgs = get_vkitti_files(self.semantic_root, exclude, "png")

        # get ids and shuffle
        self.ids = list(sorted(self.coco.imgs.keys()))
        if cfg.VKITTI_DATASET.SHUFFLE:
            print("Shuffling samples")
            random.Random(4).shuffle(self.ids)

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

        # self.depth_imgs = get_vkitti_files(self.depth_root, exclude, "png")

        self.transforms = transforms

        if cfg.VKITTI_DATASET.MAX_SAMPLES != None:
            self.ids = self.ids[:cfg.VKITTI_DATASET.MAX_SAMPLES]


    # def find_k_nearest(self, lidar_fov):
    #     k_number = self.cfg.VKITTI_DATASET.DEPTH.K
    #     b_lidar_fov = torch.unsqueeze(lidar_fov, dim=0)

    #     distances = torch.cdist(b_lidar_fov, b_lidar_fov, p=2)
    #     _, indices = torch.topk(distances, k_number + 1, dim=2, largest=False)
    #     indices = indices[:, :, 1:]  # B x N x 3

    #     return indices.squeeze_(0).long()

    # def sample_depth_img(self, depth_tensor):
    #     (img_height, img_width) = depth_tensor.shape

    #     N_num = np.floor(img_height*img_width*self.cfg.VKITTI_DATASET.DEPTH.SPARSITY)
    #     N_num = int(N_num)

    #     rand_x_coors = random.choices(range(0,img_width-1), k=N_num)
    #     rand_y_coors = random.choices(range(0,img_height-1), k=N_num)

    #     coors = torch.zeros((N_num, 2))

    #     # coors in the form of NxHxW
    #     coors[:, 1] = torch.tensor(rand_x_coors, dtype=torch.long)
    #     coors[:, 0] = torch.tensor(rand_y_coors, dtype=torch.long)
    #     coors = coors.long()

    #     # find unique coordinates
    #     _, indices = torch.unique(coors[:, :2], dim=0, return_inverse=True)
    #     unique_indices = torch.zeros_like(torch.unique(indices))

    #     current_pos = 0
    #     for i, val in enumerate(indices):
    #         if val not in indices[:i]:
    #             unique_indices[current_pos] = i
    #             current_pos += 1

    #     # Make sure there are a total of N_num depth points
    #     diff = len(unique_indices) - N_num
    #     iter = 1
    #     while diff < 0:
    #         step = 1000
    #         rand_x_coors.extend(random.choices(range(0,img_width-1), k=step))
    #         rand_y_coors.extend(random.choices(range(0,img_height-1), k=step))
            
    #         coors = torch.zeros((N_num + step*iter, 2))

    #         # coors in the form of NxHxW
    #         coors[:, 1] = torch.tensor(rand_x_coors, dtype=torch.long)
    #         coors[:, 0] = torch.tensor(rand_y_coors, dtype=torch.long)
    #         coors = coors.long()
    #         _, indices = torch.unique(coors[:, :2], dim=0, return_inverse=True)
    #         unique_indices = torch.zeros_like(torch.unique(indices))

    #         current_pos = 0
    #         for i, val in enumerate(indices):
    #             if val not in indices[:i]:
    #                 unique_indices[current_pos] = i
    #                 current_pos += 1
            
    #         diff = len(unique_indices) - N_num
    #         iter = iter +1

    #     imPts = coors[unique_indices]
    #     depth = depth_tensor[imPts[:, 0], imPts[:, 1]]/256

    #     inds = depth < self.cfg.VKITTI_DATASET.DEPTH.MAX_DEPTH

    #     return imPts[inds, :][:N_num], depth[inds][:N_num]

    # def sample_depth_gt_img(self, depth_tensor):
    #     (img_height, img_width) = depth_tensor.shape

    #     N_num = np.floor(img_height*img_width*self.cfg.VKITTI_DATASET.DEPTH.SPARSITY)*10
    #     N_num = int(N_num)
    #     rand_x_coors = random.choices(range(0,img_width-1), k=N_num)
    #     rand_y_coors = random.choices(range(0,img_height-1), k=N_num)

    #     coors = torch.zeros((N_num, 2))

    #     # coors in the form of NxHxW
    #     coors[:, 1] = torch.tensor(rand_x_coors, dtype=torch.long)
    #     coors[:, 0] = torch.tensor(rand_y_coors, dtype=torch.long)
    #     coors = coors.long()

    #     # find unique coordinates
    #     _, indices = torch.unique(coors[:, :2], dim=0, return_inverse=True)
    #     unique_indices = torch.zeros_like(torch.unique(indices))

    #     current_pos = 0
    #     for i, val in enumerate(indices):
    #         if val not in indices[:i]:
    #             unique_indices[current_pos] = i
    #             current_pos += 1

    #     imPts = coors[unique_indices]

    #     depth = depth_tensor[imPts[:, 0], imPts[:, 1]]/256

    #     # filter out long ranges of depth
    #     inds = depth < self.cfg.VKITTI_DATASET.DEPTH.MAX_DEPTH

    #     return imPts[inds, :][:N_num], depth[inds][:N_num]

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

        scene = img_filename.split("/")[-6]
        weather = img_filename.split("/")[-5]
        basename = img_filename.split(".")[-2].split("_")[-1]

        # TODO: semantic rgb to z-index
        semantic_img_filename = [s for s in self.semantic_imgs if (
            scene in s and basename in s and weather in s)][0]
        
        semantic_mask = Image.open(semantic_img_filename)
        # self.mask_to_class(np.asarray(semantic_mask))
        semantic_mask = self.mask_to_class(np.asarray(semantic_mask))
        semantic_mask = np.asarray(semantic_mask, dtype=np.long)
        
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
            #TODO: check whether or not to add background 0 label
            labels.append(self.obj_categories.index(label))
            area = coco_annotation[i]['area']
            areas.append(area)

            iscrowd.append(coco_annotation[i]['iscrowd'])

            category_ids.append(category_id)

        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Iscrowd

        category_ids = torch.as_tensor(category_ids, dtype=torch.int64)

        # Num of instance objects
        num_objs = torch.as_tensor(num_objs, dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        my_annotation["category_ids"] = category_ids
        my_annotation["num_instances"] = num_objs
        my_annotation['masks'] = masks

        my_annotation["semantic_mask"] = semantic_mask
        
        return img_filename, my_annotation

    def __getitem__(self, index):

        img_filename, ann = self.get_coco_ann(index)

        # scene = img_filename.split("/")[-6]
        # weather = img_filename.split("/")[-5]
        basename = img_filename.split(".")[-2].split("_")[-1]

        img_filename = os.path.join(self.cfg.VKITTI_DATASET.DATASET_PATH.ROOT, img_filename)
        source_img = np.asarray(Image.open(img_filename))
        image_id = ann["image_id"]

        # depth_filename = [s for s in self.depth_imgs if (
        #     scene in s and basename in s and weather in s)][0]
        
        # depth_img = np.asarray(Image.open(depth_filename))
        

        if self.transforms is not None:
            
            transformed = self.transforms(
                image=source_img,
                masks=[*ann["masks"], ann['semantic_mask']],
                bboxes=ann["boxes"],
                class_labels=ann["labels"]
            )
            
        
            source_img = transformed["image"]
            # depth_img = transformed["masks"][-1]
            # print("semantic_Gt unique: ", torch.unique(transformed['masks'][-2]))
            semantic_mask = np.asarray(transformed['masks'][-1].cpu().numpy(), dtype=np.long)

            num_boxes = len(transformed['bboxes'])
            instance = Instances(semantic_mask.shape)
            
            if num_boxes:
                instance_masks =[mask.cpu().numpy() for mask in transformed['masks'][:num_boxes]]
                instance.gt_masks = BitMasks(np.asarray(instance_masks))
                instance.gt_classes = torch.as_tensor(transformed['class_labels'])
                instance.gt_boxes = Boxes(transformed['bboxes'])
            else:
                instance.gt_masks = BitMasks(torch.Tensor([]).view(0,1,1))
                instance.gt_classes = torch.as_tensor([])
                instance.gt_boxes = Boxes([])

        # sparse_depth_gt_full = np.array(depth_img, dtype=int).astype(np.float)/256
        # sparse_depth_gt_full = torch.from_numpy(sparse_depth_gt_full)
        # sparse_depth_gt_full = torch.where(sparse_depth_gt_full >= self.cfg.VKITTI_DATASET.DEPTH.MAX_DEPTH, 
        #     torch.tensor([0], dtype=torch.float64), 
        #     sparse_depth_gt_full)

        # imPts, depth = self.sample_depth_img(depth_img)
        # imPts_gt, depth_gt = self.sample_depth_gt_img(depth_img)

        # virtual_lidar = torch.zeros(imPts.shape[0], 3)
        # virtual_lidar[:, 0:2] = imPts
        # virtual_lidar[:, 2] = depth

        # mask = torch.zeros(source_img.shape[1:], dtype=torch.bool)
        # mask[imPts[:, 0], imPts[:, 1]] = True

        # k_nn_indices = self.find_k_nearest(virtual_lidar)

        # sparse_depth = torch.zeros_like(
        #     source_img[0, :, :].unsqueeze_(0), dtype=torch.float)


        # sparse_depth[0, imPts[:, 0], imPts[:, 1]
        #                 ] = depth.clone().detach().type(torch.float)

        # -------Generate virtual ground truth

        # sparse_depth_gt = torch.zeros_like(
        #     source_img[0, :, :].unsqueeze_(0), dtype=torch.float)

        # sparse_depth_gt[0, imPts[:, 0], imPts[:, 1]
        #                 ] = depth.clone().detach().type(torch.float)

        # sparse_depth_gt[0, imPts_gt[:, 0], imPts_gt[:, 1]
        #                 ] = depth_gt.clone().detach().type(torch.float)

        return {
            "image": source_img, 
            "semantic": semantic_mask,
            'instance': instance, 
            "image_id": image_id,
            "basename": basename,
            "n_instances": num_boxes
            # "virtual_lidar": virtual_lidar, 
            # "mask": mask, 
            # "sparse_depth": sparse_depth, 
            # "k_nn_indices": k_nn_indices, 
            # "sparse_depth_gt": sparse_depth_gt, 
            # "sparse_depth_gt_full": sparse_depth_gt_full  
        }
        

    def __len__(self):
        return len(self.ids)

def get_transform(cfg):

    custom_transforms = []
    h_crop, w_crop = (cfg.VKITTI_DATASET.RANDOMCROP.HEIGHT, cfg.VKITTI_DATASET.RANDOMCROP.WIDTH)
    h_ccrop, w_ccrop = (cfg.VKITTI_DATASET.CENTER_CROP.HEIGHT, cfg.VKITTI_DATASET.CENTER_CROP.WIDTH)
    h_resize, w_resize = (cfg.VKITTI_DATASET.RESIZE.HEIGHT, cfg.VKITTI_DATASET.RESIZE.WIDTH)


    # custom_transforms.append(A.Rotate(limit=int(15), p=1))
    custom_transforms.append(A.Resize(height=h_resize, width=w_resize))
    custom_transforms.append(A.RandomCrop(width=w_crop, height=h_crop))
    custom_transforms.append(A.CenterCrop(width=w_ccrop, height=h_ccrop))
    # custom_transforms.append(A.HorizontalFlip(p=cfg.VKITTI_DATASET.HFLIP))
    custom_transforms.append(A.Normalize(mean=cfg.VKITTI_DATASET.NORMALIZE.MEAN, std=cfg.VKITTI_DATASET.NORMALIZE.STD))
    custom_transforms.append(ToTensorV2())

    return A.Compose(custom_transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


def get_datasets(cfg):
    
    vkitti_dataset = vkittiDataset(cfg, get_transform(cfg))

    if cfg.VKITTI_DATASET.SPLIT_DATASET:

        train_size, val_size, test_size = cfg.VKITTI_DATASET.SPLITS

        if train_size + val_size + test_size > 1:
            raise AssertionError("split sizes must add up to 1.0")

        len_val = math.floor(len(vkitti_dataset)*val_size)
        len_test = math.floor(len(vkitti_dataset)*test_size)

        if train_size + val_size + test_size == 1:
            len_train = len(vkitti_dataset) - len_val - len_test
        else:
            len_train = math.floor(len(vkitti_dataset)*train_size)

        if len_train < 1 or len_val < 1:
            raise AssertionError("datasets length cannot be zero")

        train_set, val_set, test_set = random_split(vkitti_dataset, [len_train, len_val, len_test], generator=torch.Generator().manual_seed(42))
        print("Training set: {} samples".format(len_train))
        print("Evaluating set: {} samples".format(len_val))
        print("Test set: {} samples".format(len_test))
        return train_set, val_set, test_set
    else:
        print("{} samples on this dataset".format(len(vkitti_dataset)))
        return vkitti_dataset

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
            'basename': [i['basename'] for i in batch]
            # 'virtual_lidar': [i['virtual_lidar'] for i in batch],
            # 'mask': [i['mask'] for i in batch],
            # 'sparse_depth': [i['sparse_depth'] for i in batch],
            # 'k_nn_indices': [i['k_nn_indices'] for i in batch],
            # 'sparse_depth_gt': [i['sparse_depth_gt'] for i in batch],
            # 'sparse_depth_gt_full': [i['sparse_depth_gt_full'] for i in batch]
        }
    return collate_fn
# def collate_fn(batch):
#     return {
#         'image': torch.stack([i['image'] for i in batch]),
#         'semantic': torch.as_tensor(np.asarray([i['semantic'] for i in batch])),
#         'instance': [i['instance'] for i in batch],
#         'image_id': [i['image_id'] for i in batch],
#         'basename': [i['basename'] for i in batch]
#         # 'virtual_lidar': [i['virtual_lidar'] for i in batch],
#         # 'mask': [i['mask'] for i in batch],
#         # 'sparse_depth': [i['sparse_depth'] for i in batch],
#         # 'k_nn_indices': [i['k_nn_indices'] for i in batch],
#         # 'sparse_depth_gt': [i['sparse_depth_gt'] for i in batch],
#         # 'sparse_depth_gt_full': [i['sparse_depth_gt_full'] for i in batch]
#     }

def test_dataset(cfg, dataset, item):
    sample = dataset.__getitem__(item)
    basename = sample["basename"]
    image_id = sample["image_id"]
    source_img = sample["image"]
    # virtual_lidar = sample["virtual_lidar"]
    # mask = sample["mask"]
    # sparse_depth = sample["sparse_depth"]
    # k_nn_indices = sample["k_nn_indices"]
    # sparse_depth_gt = sample["sparse_depth_gt"]
    # sparse_depth_gt_full = sample["sparse_depth_gt_full"]
    # basename = sample["basename"]
    semantic_mask = sample["semantic"]
    instance = sample["instance"]

    save_image(source_img, os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR, "dataset_test_v2", "source_img_{}.png".format(basename)))
    # save_image(sparse_depth, os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR, "sparse_depth.png"), normalize=True)
    # save_image(sparse_depth_gt, os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR, "sparse_depth_gt.png"), normalize=True)
    # save_image(sparse_depth_gt_full, os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR, "sparse_depth_gt_full.png"), normalize=True)

    
    # cv2.imwrite(os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR, "semantic_mask_cv2.png"), semantic_mask)

    plt.imshow(semantic_mask)
    plt.savefig(os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR, "dataset_test_v2", "semantic_mask_{}.png".format(basename)))
    # plt.imshow(mask.cpu().numpy())
    # plt.savefig(os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR, "mask.png"))
   

    # Visualize annotations
    image = source_img.cpu().numpy().transpose((1, 2, 0))*255
    image = visualize_masks(instance.get("gt_masks"))
    image = visualize_bboxes(image, instance.get("gt_boxes"))
    cv2.imwrite(os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR, "dataset_test_v2", "masks_{}.png".format(basename)), image)


def get_dataloaders(cfg):

    batch_size = cfg.BATCH_SIZE

    if cfg.VKITTI_DATASET.SPLIT_DATASET:
        train_set, val_set, test_set = get_datasets(cfg)
        # test_dataset(cfg, train_set, 0)
        # for i in range(96):
        #     test_dataset(cfg, train_set, i)

        train_collate_fn = collate_fn_wrapper(train_set)
        data_loader_train = torch.utils.data.DataLoader(train_set,
                                                        batch_size=batch_size,
                                                        num_workers=40,
                                                        collate_fn=train_collate_fn,
                                                        drop_last=False)
        val_collate_fn = collate_fn_wrapper(val_set)
        data_loader_val = torch.utils.data.DataLoader(val_set,
                                                      batch_size=batch_size,
                                                      num_workers=40,
                                                      collate_fn=val_collate_fn,
                                                      drop_last=False)
        test_collate_fn = collate_fn_wrapper(test_set)
        data_loader_test = torch.utils.data.DataLoader(test_set,
                                                      batch_size=batch_size,
                                                      num_workers=40,
                                                      collate_fn=test_collate_fn,
                                                      drop_last=False)

        return data_loader_train, data_loader_val, data_loader_test

    else:
        dataset = get_datasets(cfg)
        collate_fn = collate_fn_wrapper(dataset)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  num_workers=40,
                                                  collate_fn=collate_fn,
                                                  drop_last=False)
    return data_loader, None, None


#
