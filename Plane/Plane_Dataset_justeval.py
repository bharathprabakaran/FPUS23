from __future__ import print_function, division
from PIL import Image
import xml.etree.ElementTree as ET
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class orientation_Dataset(Dataset):


    def __init__(self, img_path, transform=transforms.ToTensor()):
        """
        Args:
            img_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_path = img_path
        self.transform = transform

        self.X = []
        self.Y = []
        self.X1 = []
        self.Y1 = []
        self.X2 = []
        self.Y2 = []
        self.X3 = []
        self.Y3 = []
        self.X4 = []
        self.Y4 = []
        hdvb = 0
        hdvf = 0
        huvb = 0
        huvf = 0

        # Scan all xml files for labels and image names
        for obj in os.listdir(img_path):
            x = os.path.join(img_path + obj )
            #print(x)
            self.X1.append(x)


        self.X = self.X1



    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # read images from files
        img_name = self.X[idx]
        image = Image.open(img_name)

        image = np.array(image)
        image = self.transform(image)
        # define label of the image


        sample = (image)

        return sample

#image_path = '/home/erik/PycharmProjects/Test/unet-lightning/dataset/US/four_poses/stream_hdvb_aroundabd_h/'
#orientation_Dataset(image_path)