# Basic python and ML Libraries
import os
import random
import numpy as np
import pandas as pd
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')
import torch.nn.utils.prune as prune
# We will be reading images using OpenCV
import cv2
from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
# xml library for parsing xml files
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

from OD_US_Dataset import OD_US_Dataset

# defining the files directory and testing directory
image_path = '../Dataset/four_poses/' #'/srv/data/eostrowski/Dataset/four_poses/'
path = '../Dataset/boxes/annotation/' #'/srv/data/eostrowski/Dataset/boxes/annotation/'


# Basic module in Resnet architecture
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.add_relu = torch.nn.quantized.FloatFunctional()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        
        
        out = self.add_relu.add_relu(out, identity)
        return out


# Build Resnet with the help of the basic block class
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=5):
        super().__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.quant(x)
        x = self.conv1(x)  # 224x224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 112x112

        x = self.layer1(x)  # 56x56
        x = self.layer2(x)  # 28x28
        x = self.layer3(x)  # 14x14
        x = self.layer4(x)  # 7x7

        
        x = self.dequant(x)
        return x


# Define final Resnet depth (e.g. Resnet 8 or 34)
def resnet34():
    print('################# RESNET10 BACKBONE ###########################')
    #layers=[3, 4, 6, 3]
    layers = [1, 1, 1, 1]
    
    model = ResNet(BasicBlock, layers)
    return model

# check dataset
dataset = OD_US_Dataset(path, image_path,)
print('length of dataset = ', len(dataset), '\n')


# Define Faster-RCNN with our Resnet backbone
def get_object_detection_model(num_classes=5):
   

    backbone = resnet34()

    
    backbone.out_channels = 512

    
    anchor_generator = AnchorGenerator(sizes=((128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
    return model



# Define dataset transforms/ converting the images to tensors
def get_transform(train):
    if train:
        return A.Compose([
            
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


# use our dataset and defined transformations
dataset = OD_US_Dataset(path, image_path, transforms= get_transform(train=True))
dataset_test = OD_US_Dataset(path, image_path, transforms= get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

# train test split
test_split = 0.2
tsize = int(len(dataset)*test_split)
dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=4, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)


# to train on gpu if selected.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


num_classes = 5

# get the model using our helper function
model = get_object_detection_model(num_classes)

#Load model
model.load_state_dict(torch.load('../saved_models/OD_10/quant_model.pth', map_location=device))




model.to(device)

# evaluate on the test dataset
evaluate(model, data_loader_test, device=device)

