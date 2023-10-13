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
            file_name = os.path.join(img_path + obj )
            print(obj)
            for n in os.listdir(file_name):

                x = os.path.join(file_name +'/'+ n)

                # Convert labels in numbers
                y = obj
                if y == 'BPD_PLANE':
                        #print(y)
                        y = (0)
                        hdvb +=1
                        self.X1.append(x)
                        self.Y1.append(y)
                        #print(name)


                elif y == 'AC_PLANE':
                        #print(y)
                        y = (1)
                        hdvf += 1
                        self.X2.append(x)
                        self.Y2.append(y)
                        #print(name)
                elif y == 'FL_PLANE':
                        #print(y)
                        y = (2)
                        huvb += 1
                        self.X3.append(x)
                        self.Y3.append(y)
                        #print(name)

                elif y == 'NO_PLANE':
                        #print(y)
                        y = (3)
                        huvf += 1
                        self.X4.append(x)
                        self.Y4.append(y)
                        #print(name)



        self.X = self.X1 + self.X2 + self.X3 + self.X4
        self.Y = self.Y1 + self.Y2 + self.Y3 + self.Y4



    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # read images from files
        img_name = self.X[idx]
        image = Image.open(img_name)

        image = np.array(image)
        image = self.transform(image)
        # define label of the image
        label = self.Y[idx]
        label = np.array(label)



        sample = (image, label)

        return sample
