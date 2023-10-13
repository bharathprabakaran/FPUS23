##Import all required objects and packages for data and DNN
from torchvision import models
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from USDataset_transform import USDataset
from torch.utils.data import random_split
from torchmetrics.functional import accuracy
from argparse import ArgumentParser
from torch.nn.functional import cross_entropy
from torch.optim import Adam
import torch.nn as nn
import os
from USDataset_transform import RandomVerticalFlip
import torch.nn.utils.prune as prune


# Data and Annotation directories
image_path = '../Dataset/four_poses/' #'/srv/data/eostrowski/Dataset/four_poses/'
path = '../Dataset/annos/annotation/' #'/srv/data/eostrowski/Dataset/annos/annotation/'
n_class = 4


######################################################################################################
# HERE PATH TO FLIPPED MODEL CHECKPOINT
#######################################################################################################
CKPT_PATH = '../saved_models/Orientation_RES8/epoch=8-step=21239.ckpt'


# basic module repeatedly used in resnet architecture
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


# Define Resnet with the help of the basic block class
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=4):
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
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
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
        #x = self.layer4(x)  # 7x7

        x = self.avgpool(x)  # 1x1
        x = torch.flatten(x, 1)  # remove 1 X 1 grid and make vector of tensor shape
        x = self.fc(x)
        
        x = self.dequant(x)
        return x


# Define explicit resnet structure (eg. Resnet34 or Resnet8)
def resnet34():
    #print('################### USING RESNET34 ###################################################')
    #layers = [3, 4, 6, 3]
    print('################### USING RESNET8 ###################################################')
    layers = [0,0,0,0]
    model = ResNet(BasicBlock, layers)
    return model


# Pretrained ResNet model
def get_model(pretrained=False):
    model = resnet34()
    trained_kernel = model.conv1.weight
    new_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :] = torch.stack([torch.mean(trained_kernel, 1)] * 3, dim=1)
    model.conv1 = new_conv
    return model

# Change first layer for dataset resolution ResNet mode
class ImageClassifier(pl.LightningModule):
    def __init__(self, model, num_classes=n_class, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        
        loss = cross_entropy(preds, y)
        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy(preds, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        
        valloss = cross_entropy(preds, y)
        self.log('val_loss', valloss)
        self.log('val_accuracy', accuracy(preds, y))
        return valloss

    def validation_epoch_end(self, validation_step_outputs):
        print(f'val_loss= {validation_step_outputs[len(validation_step_outputs)-1]:.4f}')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args():
        parser = ArgumentParser()

        parser.add_argument('--image_path',  type=str,
                            dest='image_path')#, default='/home/erik/PycharmProjects/Test/unet-lightning/dataset/US/four_poses/')
        parser.add_argument('--annotation_path', type=str,
                            dest='annotation_path')#, default='/home/erik/PycharmProjects/Test/unet-lightning/dataset/US/annos/annotation/')
        parser.add_argument('--checkpoint_path', type=str,  dest='CKPT_PATH')#, default='/home/erik/PycharmProjects/Test/unet-lightning/saved_models/epoch=0-step=375.ckpt')
        parser.add_argument('--workers', type=int, default=8, dest='num_workers')
        parser.add_argument('--batch_size', type=int, default=8, dest='batch_size')
        return parser

##############################################################################################################################################################
#Define Model

classes = ('hdvb', 'hdvf', 'huvb', 'huvf')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model()
classifier = ImageClassifier(model)

# Parse arguments and Define Hyper-parameters
parser = classifier.add_model_specific_args()
args = parser.parse_args()
num_workers = args.num_workers
batch_size = args.batch_size

# Generate Dataset from paths
train_transforms = RandomVerticalFlip()


dataset = USDataset(path, image_path, transform=None)
##############################################################################################################################################################

# Re-build Train, Test, and Validation Dataset
n_val_test = int(len(dataset) * 0.2)
n_test = int(len(dataset) * 0.1)
n_train = len(dataset) - n_val_test
test_1k = len(dataset)-(len(dataset) - 1000)
test_4k = len(dataset)-(len(dataset) - 4000)

train_ds, val_test_ds = random_split(dataset, [n_train, n_val_test], generator=torch.Generator().manual_seed(10))
_, ds_1k = random_split(dataset, [len(dataset)-1000, test_1k])
_, ds_4k = random_split(dataset, [len(dataset)-4000, test_4k])
n_val = int(n_val_test / 2)
n_test = n_val_test - n_val

val_ds , test_ds = random_split(val_test_ds, [n_val, n_test], generator=torch.Generator().manual_seed(11))

##############################################################################################################################################################

# DataLoader
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
loader_1k = torch.utils.data.DataLoader(ds_1k, batch_size=batch_size, num_workers=num_workers)
loader_4k = torch.utils.data.DataLoader(ds_4k, batch_size=batch_size, num_workers=num_workers)

# Testing
model = get_model()
model = ImageClassifier(model).load_from_checkpoint(CKPT_PATH, strict=False)





model.to(device)

model.eval()



##############################################################################################################################################################

# Test 1
# Evaluation using Test dataset

n_correct = 0
n_samples = 0
n_class_correct = [0 for i in range(4)]
n_class_samples = [0 for i in range(4)]
false_pos = [0 for i in range(4)]
false_neg = [0 for i in range(4)]
conf = [0 for i in range(4)]
conf_samples = [0 for i in range(4)]


print('#'*60)
print('Fixed Test Dataset')
print('#'*60)

for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    logits = torch.detach(outputs)
    logits = torch.sigmoid(logits)
    # max returns (value, index)

    _, predicted = torch.max(outputs, 1)
    n_samples += labels.size(0)
    n_correct += (predicted == labels).sum().item()

    for i in range(batch_size):
        if labels.shape[0] != batch_size:
            break
        label = labels[i]
        pred = predicted[i]
        n_class_samples[label] += 1

        if label == 0:
            conf[0] += logits[0][0]
            conf_samples[0] += 1
        elif label== 1:
            conf[1] += logits[0][1]
            conf_samples[1] += 1
        elif label == 2:
            conf[2] += logits[0][2]
            conf_samples[2] += 1
        elif label == 3:
            conf[3] += logits[0][3]
            conf_samples[3] += 1

        if (label == pred):
            n_class_correct[label] += 1
        else:
            if (label == 0):
                false_neg[0] += 1
            elif (label == 1):
                false_neg[1] += 1
            elif (label == 2):
                false_neg[2] += 1
            elif (label == 3):
                false_neg[3] += 1

            if (pred == 0):
                false_pos[0] += 1
            elif (pred == 1):
                false_pos[1] += 1
            elif (pred == 2):
                false_pos[2] += 1
            elif (pred == 3):
                false_pos[3] += 1

acc = 100.00 * n_correct / n_samples
n_false_pos = sum(false_pos)
n_false_neg = sum(false_neg)
rate_fp = 100.00 * n_false_pos / n_samples
rate_fn = 100.00 * n_false_neg / n_samples
print(f'Accuracy of the network: {acc} %')

print(f'Number of Errors: {n_false_pos} / {n_samples} ')

# print(f'Rate of False Negatives: {rate_fn} / Number of False Negatives: {n_false_neg} ')
# print('\n')
print(f'Confidence of the Network: {100 * (sum(conf) / n_samples):.1f} %')
print('\n')
for i in range(4):
    print(f'Number of Samples in Class {classes[i]}: {n_class_samples[i]}')
    if n_class_samples[i] != 0:
        acc = 100.00 * n_class_correct[i] / n_class_samples[i]
        rate_fp = 100.00 * false_pos[i] / n_class_samples[i]
        rate_fn = 100.00 * false_neg[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
        print(  # f'Rate of False Positives of {classes[i]}: {100*(rate_fp):.1f} % //'
            f'Number of False Positives of {classes[i]}: {false_pos[i]}')  # / {n_class_samples[i]}')
        print(  # f'Rate of False Negatives of {classes[i]}: {100*rate_fn:.1f} % //'
            f'Number of False Negatives of {classes[i]}: {false_neg[i]}')  # / {n_class_samples[i]}')
        if conf_samples[i] != 0:
            print(f'Confidence of the Network of {classes[i]}: {100 * (conf[i] / conf_samples[i]):.1f} %')
        print('\n')

##############################################################################################################################################################

# Test 2
# Evaluation using 1000 random samples from the dataset

print('#'*60)
print('1000 random Samples from Dataset')
print('#'*60)

n_correct = 0
n_samples = 0
n_class_correct = [0 for i in range(4)]
n_class_samples = [0 for i in range(4)]
false_pos = [0 for i in range(4)]
false_neg = [0 for i in range(4)]
conf = [0 for i in range(4)]
conf_samples = [0 for i in range(4)]
for images, labels in loader_1k:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    logits = torch.detach(outputs)
    logits = torch.sigmoid(logits)
    # max returns (value, index)

    _, predicted = torch.max(outputs, 1)
    n_samples += labels.size(0)
    n_correct += (predicted == labels).sum().item()

    for i in range(batch_size):
        if labels.shape[0] != batch_size:
            break
        label = labels[i]
        pred = predicted[i]
        n_class_samples[label] += 1

        if label == 0:
            conf[0] += logits[0][0]
            conf_samples[0] += 1
        elif label == 1:
            conf[1] += logits[0][1]
            conf_samples[1] += 1
        elif label == 2:
            conf[2] += logits[0][2]
            conf_samples[2] += 1
        elif label == 3:
            conf[3] += logits[0][3]
            conf_samples[3] += 1

        if (label == pred):
            n_class_correct[label] += 1
        else:
            if (label == 0):
                false_neg[0] += 1
            elif (label == 1):
                false_neg[1] += 1
            elif (label == 2):
                false_neg[2] += 1
            elif (label == 3):
                false_neg[3] += 1

            if (pred == 0):
                false_pos[0] += 1
            elif (pred == 1):
                false_pos[1] += 1
            elif (pred == 2):
                false_pos[2] += 1
            elif (pred == 3):
                false_pos[3] += 1

acc = 100.00 * n_correct / n_samples
n_false_pos = sum(false_pos)
n_false_neg = sum(false_neg)
rate_fp = 100.00 * n_false_pos / n_samples
rate_fn = 100.00 * n_false_neg / n_samples
print(f'Accuracy of the network: {acc} %')

print(f'Number of Errors: {n_false_pos} / {n_samples} ')

# print(f'Rate of False Negatives: {rate_fn} / Number of False Negatives: {n_false_neg} ')
# print('\n')
print(f'Confidence of the Network: {100 * (sum(conf) / n_samples):.1f} %')
print('\n')
for i in range(4):
    print(f'Number of Samples in Class {classes[i]}: {n_class_samples[i]}')
    if n_class_samples[i] != 0:
        acc = 100.00 * n_class_correct[i] / n_class_samples[i]
        rate_fp = 100.00 * false_pos[i] / n_class_samples[i]
        rate_fn = 100.00 * false_neg[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
        print(  # f'Rate of False Positives of {classes[i]}: {100*(rate_fp):.1f} % //'
            f'Number of False Positives of {classes[i]}: {false_pos[i]}')  # / {n_class_samples[i]}')
        print(  # f'Rate of False Negatives of {classes[i]}: {100*rate_fn:.1f} % //'
            f'Number of False Negatives of {classes[i]}: {false_neg[i]}')  # / {n_class_samples[i]}')
        if conf_samples[i] != 0:
            print(f'Confidence of the Network of {classes[i]}: {100 * (conf[i] / conf_samples[i]):.1f} %')
        print('\n')

##############################################################################################################################################################

# Test 3
# Evaluation using 4000 random samples from the dataset

print('#'*60)
print('4000 random Samples from Dataset')
print('#'*60)

n_correct = 0
n_samples = 0
n_class_correct = [0 for i in range(4)]
n_class_samples = [0 for i in range(4)]
false_pos = [0 for i in range(4)]
false_neg = [0 for i in range(4)]
conf = [0 for i in range(4)]
conf_samples = [0 for i in range(4)]
for images, labels in loader_4k:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    logits = torch.detach(outputs)
    logits = torch.sigmoid(logits)
    # max returns (value, index)

    _, predicted = torch.max(outputs, 1)
    n_samples += labels.size(0)
    n_correct += (predicted == labels).sum().item()

    for i in range(batch_size):
        if labels.shape[0] != batch_size:
            break
        label = labels[i]
        pred = predicted[i]
        n_class_samples[label] += 1

        if label == 0:
            conf[0] += logits[0][0]
            conf_samples[0] += 1
        elif label == 1:
            conf[1] += logits[0][1]
            conf_samples[1] += 1
        elif label == 2:
            conf[2] += logits[0][2]
            conf_samples[2] += 1
        elif label == 3:
            conf[3] += logits[0][3]
            conf_samples[3] += 1

        if (label == pred):
            n_class_correct[label] += 1
        else:
            if (label == 0):
                false_neg[0] += 1
            elif (label == 1):
                false_neg[1] += 1
            elif (label == 2):
                false_neg[2] += 1
            elif (label == 3):
                false_neg[3] += 1

            if (pred == 0):
                false_pos[0] += 1
            elif (pred == 1):
                false_pos[1] += 1
            elif (pred == 2):
                false_pos[2] += 1
            elif (pred == 3):
                false_pos[3] += 1

acc = 100.00 * n_correct / n_samples
n_false_pos = sum(false_pos)
n_false_neg = sum(false_neg)
rate_fp = 100.00 * n_false_pos / n_samples
rate_fn = 100.00 * n_false_neg / n_samples
print(f'Accuracy of the network: {acc} %')

print(f'Number of Errors: {n_false_pos} / {n_samples} ')

# print(f'Rate of False Negatives: {rate_fn} / Number of False Negatives: {n_false_neg} ')
# print('\n')
print(f'Confidence of the Network: {100 * (sum(conf) / n_samples):.1f} %')
print('\n')
for i in range(4):
    print(f'Number of Samples in Class {classes[i]}: {n_class_samples[i]}')
    if n_class_samples[i] != 0:
        acc = 100.00 * n_class_correct[i] / n_class_samples[i]
        rate_fp = 100.00 * false_pos[i] / n_class_samples[i]
        rate_fn = 100.00 * false_neg[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
        print(  # f'Rate of False Positives of {classes[i]}: {100*(rate_fp):.1f} % //'
            f'Number of False Positives of {classes[i]}: {false_pos[i]}')  # / {n_class_samples[i]}')
        print(  # f'Rate of False Negatives of {classes[i]}: {100*rate_fn:.1f} % //'
            f'Number of False Negatives of {classes[i]}: {false_neg[i]}')  # / {n_class_samples[i]}')
        if conf_samples[i] != 0:
            print(f'Confidence of the Network of {classes[i]}: {100 * (conf[i] / conf_samples[i]):.1f} %')
        print('\n')
