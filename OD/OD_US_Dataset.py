# Basic python and ML Libraries
import os
import random
import numpy as np
import pandas as pd
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2
from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from xml.etree import ElementTree as ET

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# these are the helper libraries imported.
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import torch.nn as nn
# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class OD_US_Dataset(torch.utils.data.Dataset):

    def __init__(self, ano_path, img_path, transforms=None):

        # classes: 0 index is reserved for background
        self.classes = ['bkg', 'Head', 'Abdomen', 'Arms', 'Legs']

        self.ano_path = ano_path
        self.img_path = img_path

        self.transforms = transforms
        self.bbox = []
        self.X = []
        self.Y = []
        self.la = []
        self.LA = []

        # Scan all xml files for labels and image names
        for obj in os.listdir(ano_path):
            file_name = os.path.join(ano_path + obj + '/annotations.xml')
            dom = ET.parse(file_name)

            names = dom.findall('image')
            for n in names:
                self.bbox = []
                self.la = []
                name = n.attrib.get('name')
                lab = n.findall('box')
                if not (lab == []):
                    for l in lab:
                        xtl = l.attrib.get('xtl')
                        ytl = l.attrib.get('ytl')
                        xbr = l.attrib.get('xbr')
                        ybr = l.attrib.get('ybr')
                        label = l.attrib.get('label')
                        box = [xtl, ytl, xbr, ybr]

                        if label == 'head':
                            self.la.append(1)
                        elif label == 'abdomen':
                            self.la.append(2)
                        elif label == 'arm':
                            self.la.append(3)
                        elif label == 'legs':
                            self.la.append(4)

                        self.bbox.append(box)

                    x = os.path.join(img_path + obj + '/' + name)
                    self.Y.append(self.bbox)
                    self.X.append(x)
                    self.LA.append(self.la)





    def __getitem__(self, idx):

        img_name = self.X[idx]


        # reading the images and converting them to correct size and color
        img = cv2.imread(img_name)
        img_res = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # diving by 255
        img_res /= 255.0

        # cv2 image gives size as height x width
        wt = img.shape[1]
        ht = img.shape[0]

        labels = self.LA[idx]
        boxes = self.Y[idx]
        boxes = np.array(boxes).astype(float)
        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        #print(labels)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image=img_res,
                                     bboxes=target['boxes'],
                                     labels=labels)

            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return img_res, target

    def __len__(self):
        return len(self.X)

