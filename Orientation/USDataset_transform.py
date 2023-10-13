from __future__ import print_function, division
from PIL import Image
import xml.etree.ElementTree as ET
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random
#random.seed(0)
import cv2

# class to filp image along the y-axis
class RandomVerticalFlip:
    def __init__(self):
        pass

    def __call__(self, image):
        return image.transpose(Image.FLIP_LEFT_RIGHT)

        #return image


class USDataset(Dataset):


    def __init__(self, ano_path, img_path, transform=transforms.ToTensor()):#, flip= True):
        """
        Args:
            ano_path (string): Path to the xml file with annotations.
            img_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ano_path = ano_path
        self.img_path = img_path
        self.transform = transform
        #self.flip = flip
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
        for obj in os.listdir(ano_path):
            file_name = os.path.join(ano_path + obj + '/annotations.xml')
            dom = ET.parse(file_name)

            names = dom.findall('image')
            for n in names:
                name = n.attrib.get('name')
                #print(name)


                # Full path to the Images
                x = os.path.join(img_path + obj + '/' + name)

                # Convert labels in numbers
                label = n.findall('tag/attribute')

                for label in label:
                    y = label.text
                    #print(y)
                    #print('#############')
                    if y == 'hdvb':
                        y = (0)
                        hdvb +=1
                        self.X1.append(x)
                        self.Y1.append(y)
                    elif y == 'hdvf':
                        y = (1)
                        hdvf += 1
                        self.X2.append(x)
                        self.Y2.append(y)
                    elif y == 'huvb':
                        y = (2)
                        huvb += 1
                        self.X3.append(x)
                        self.Y3.append(y)
                    elif y == 'huvf':
                        y = (3)
                        huvf += 1
                        self.X4.append(x)
                        self.Y4.append(y)



        self.X = self.X1 + self.X2 + self.X3 + self.X4
        self.Y = self.Y1 + self.Y2 + self.Y3 + self.Y4



    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # read images and transform them to tensors
        img_name = self.X[idx]
        image = Image.open(img_name)

        #print(f'################ TEST SHAPE : {test.shape}')
        if not (self.transform is None):
            image = self.transform(image)
            #print(f'####### NUMBA OF FLIPS: {i}')

            image = np.array(image)

        #print(f'####### NUMBA OF FLIPS: {i}')


        tensor_transform = transforms.ToTensor()
        image = tensor_transform(image)


        # add label to output
        label = self.Y[idx]
        label = np.array(label)



        sample = (image, label)

        return sample

