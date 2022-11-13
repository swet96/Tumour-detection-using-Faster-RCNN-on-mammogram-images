import os
import os.path as osp
import csv
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
import transforms as img_box_T
import torchvision.transforms as T

import albumentations as Albu
import albumentations.augmentations.transforms as Albu_T
#custom
from utils import check_bbox

#mean and std of DDSM data
# mean:  tensor([0.3204, 0.3204, 0.3204])
# std:  tensor([0.2557, 0.2557, 0.2557])

# import argparse
# parser = argparse.ArgumentParser()

# parser.add_argument('--train_img_dir', type=str, default='/home/sweta/scratch/FASTER_RCNN/DDSM_DATA2/Train/images', help="directory where the training images are present")
# parser.add_argument('--train_annotate_dir', type=str, default='/home/sweta/scratch/FASTER_RCNN/DDSM_DATA2/Train/Annotations', help="directory where the training annotations are present")
# args = parser.parse_args()
# pprint(vars(args))

class DDSM_dataset(Dataset):
    def __init__(self, img_dir, annotate_dir, height = 640, width= 640, which_transform = 'Train', full_data= False):
        self.img_dir = img_dir
        self.annotate_dir = annotate_dir
        # to resize the images to this height, in DDSM, the height and width are fixed and are 640 
        self.img_height, self.img_width = height, width 
        self.which_transform = which_transform 
        annotate_fnames = os.listdir(self.annotate_dir)
        
        if full_data:
            img_fnames= os.listdir(self.img_dir)
        else:
            img_fnames =[]
            for idx, annotate_fname in enumerate(annotate_fnames):
                annotate_path = osp.join(annotate_dir, annotate_fname)
                if osp.exists(annotate_path):
                    img_fname= annotate_fname[:-4]+'.jpg'
                    img_fnames.append(img_fname)
        self.img_fnames = img_fnames

        self.train_transform = img_box_T.Compose([img_box_T.RandomHorizontalFlip(),] )
        # A.Flip(0.5), A.RandomRotate90(0.5), A.MotionBlur(p=0.2), A.MedianBlur(blur_limit=3, p=0.1),
        # A.Blur(blur_limit=3, p=0.1),
        # self.test_transform = img_box_T.Compose([T.ToTensor(),] )

        if self.which_transform == "Train":
            self.transform = self.train_transform
        # elif which_transform == 'Test':
        #     self.transform = self.test_transform
        # elif which_transform == 'Validation':
        #     self.transform = self.test_transform
        self.common_transform=  Albu.Compose([Albu_T.CLAHE(always_apply=True),Albu_T.ColorJitter(), Albu_T.Equalize()] )
        # ,Albu_T.Emboss()
    def __getitem__(self, idx):

        # from PIL import Image
        img_fname = self.img_fnames[idx]
        img_path = osp.join(self.img_dir, img_fname)
        annotate_fname = img_fname[:-4]+'.txt'
        annotate_path = osp.join(self.annotate_dir, annotate_fname)
        # iii = Image.open(img_path)
        # print("width: ", w)
        # print("height: ", h)
        # print(np.array(iii).shape)
        # read the image,   # convert BGR to RGB color format
        img = cv2.imread(img_path) # numpy array (H,W,C)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img/= 255.0
        img = T.ToTensor()(img)
        # img = T.functional.adjust_gamma(img, 0.5)
        boxes = []
        labels = []
        area= []
        iscrowd= []

        if not osp.exists(annotate_path):
            boxes.append([]); labels.append(1); area.append(0); iscrowd.append(0)
            
        else:
            with open(annotate_path, "r", newline="") as f:
                reader = csv.reader(f)
                for line in reader:
                    # print(line) 
                    xmin = float(line[0].split(' ')[1]); ymin = float(line[0].split(' ')[2])
                    xmax = float(line[0].split(' ')[3]); ymax = float(line[0].split(' ')[4])

                    # xmin_normalised = (xmin/self.img_width); ymin_normalised = (ymin/self.img_height)
                    # xmax_normalised = (xmax/self.img_width); yamx_normalised = (ymax/self.img_height)

                    # xmin_resized = (xmin/image_width)*self.width; ymin_resized = (ymin/image_height)*self.height
                    # xmax_resized = (xmax/image_width)*self.width; yamx_resized = (ymax/image_height)*self.height
                    # box= [xmin, ymin, xmax, ymax]
                    box = [xmin, ymin, xmax, ymax ]
                    # box= check_bbox(box)
                    boxes.append(box);labels.append(1); area.append((xmax-xmin)*(ymax-ymin)); iscrowd.append(0)
                
        boxes = torch.as_tensor(boxes, dtype=torch.float32) #TODO is it needed?
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype= torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype= torch.int64)

        #use 'boxes', 'labels', 'area as the key of the dict targets '
        targets = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor(idx), 'iscrowd':iscrowd, 'area': area}

        if self.which_transform == 'Train': 
            img, targets = self.transform(img, targets)
        return img, targets

    def __len__(self):
        return len(self.img_fnames)

# d=DDSM_dataset(args.train_img_dir, args.train_annotate_dir)
# d.__getitem__(5)
#For getting the mean and std of the DDSM data
# from torch.utils.data import DataLoader
# import torchvision.transforms as T
# class MyDataset(Dataset):
#     def __init__(self, img_dir='/home/sweta/scratch/datasets/DDSM_DATA2/Train/images' , annotate_dir='/home/sweta/scratch/datasets/DDSM_DATA2/Train/Annotations' ):
#         self.img_dir = img_dir
#         self.annotate_dir = annotate_dir
#         annotate_fnames = os.listdir(self.annotate_dir)
        
#         img_fnames =[]
#         for idx, annotate_fname in enumerate(annotate_fnames):
#             annotate_path = osp.join(annotate_dir, annotate_fname)
#             if osp.exists(annotate_path):
#                 img_fname= annotate_fname[:-4]+'.jpg'
#                 img_fnames.append(img_fname)
#         self.img_fnames = img_fnames
        
#     def __getitem__(self, index):
#         img_fname = self.img_fnames[index]
#         img_path = osp.join(self.img_dir, img_fname)
#         img = cv2.imread(img_path) # numpy array (H,W,C)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
#         img/= 255.0

#         x = T.ToTensor()(img)
#         return x

#     def __len__(self):
#         return len(self.img_fnames)
    
# def get_mean_std():
    # dataset = MyDataset()
    # loader = DataLoader(dataset,batch_size=10,num_workers=1,shuffle=False)
    # mean = 0.
    # std = 0.
    # nb_samples = 0.
    # for data in loader:
    #     batch_samples = data.size(0)
    #     data = data.view(batch_samples, data.size(1), -1)
    #     # print("data: ", data.shape)
    #     # print(data.shape)
    #     mean += data.mean(2).sum(0)
    #     # print(data.mean(2).shape)
    #     std += data.std(2).sum(0)
    #     nb_samples += batch_samples
    #     # break

    # mean /= nb_samples
    # std /= nb_samples
    # return mean, std

# data = MyDataset()
# mean, std = get_mean_std()
# print("mean: ", mean)
# print("std: ", std)

# mean:  tensor([0.3204, 0.3204, 0.3204])
# std:  tensor([0.2557, 0.2557, 0.2557])