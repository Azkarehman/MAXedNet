import os
import numpy as np
import scipy.io as sio
import pdb
import time
from os.path import isfile, join

import nibabel as nib
from PIL import Image
from medpy.metric.binary import dc,hd
import skimage.transform as skiTransf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

def load_nii(imageFileName, printFileNames):
    if printFileNames == True:
        print (" ... Loading file: {}".format(imageFileName))

    img_proxy = nib.load(imageFileName)
    imageData = img_proxy.get_data()

    return (imageData, img_proxy)


def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
        imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]

    imageNames.sort()

    return imageNames

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()


class DataPrep(Dataset):
    def __init__(self, root, ind):
        self.root = root
        self.total_data = np.load(self.root, allow_pickle=True)
        #self.total_data = self.total_data[0:100] # 0-100 for DDTI and 100-200 for TN3K
        self.train_data = []
        v = []
        
        for i in range(10):
            if i == ind:
                print('t: ', i)
                v = self.total_data[i*63:(i+1)*63]
            else:
                z = self.total_data[i*63:(i+1)*63]
                for s in z:
                    image, mask = s
                    self.train_data.append([image, mask])
        
        self.train_data = np.array(self.train_data)
        print('train size: ', self.train_data.shape)
        
        self.transform = transforms.Compose([
                transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.train_data)

    def Random_augmentation(self, image, mask):
        rnum = np.random.randint(0, 4)
        
        if rnum == 0:
            image = np.flip(image, 1)
            mask = np.flip(mask, 1)
        if rnum == 1:
            image = np.flipud(image)
            mask =  np.flipud(mask)
        if rnum == 2:
            image = np.flip(image, 1)
            mask = np.flip(mask, 1)
            image = np.flipud(image)
            mask =  np.flipud(mask)
        
        return image, mask

    def __getitem__(self, index):
        image_path, mask_path = self.train_data[index]
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
         
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation = cv2.INTER_NEAREST)
        image = image[:, :, 0]
        mask = mask[:, :, 0]
        #print('before mask shape: ', mask.shape, ' ', np.unique(mask))
        
        image, mask = self.Random_augmentation(image, mask)
        imagec = image.copy()
        maskc = mask.copy()
        
        maskc[maskc == 255] = 1
        imagec = imagec/np.max(imagec)

        imagec = self.transform(imagec)
        #print('image shape: ', image.shape)
        #print('mask shape: ', maskc.shape, ' ', np.unique(maskc))
        #image = torch.Tensor(image)
        maskc = torch.Tensor(maskc)
        maskc = maskc.long()
        return imagec, maskc


class DataPrepval(Dataset):
    def __init__(self, root, ind):
        self.root = root
        self.total_data = np.load(self.root, allow_pickle=True)
        #self.total_data = self.total_data[100:200] # 0-100 for DDTI and 100-200  for TN3K
        
        self.validation_data = []
        v = []
        
        for i in range(10):
            if i == ind:
                z = self.total_data[i*63:(i+1)*63]
                for s in z:
                    image, mask = s
                    self.validation_data.append([image, mask])
            else:
                print('v: ', i)
                v = self.total_data[i*63:(i+1)*63]
        
        self.validation_data = np.array(self.validation_data)
        print('validation size: ', self.validation_data.shape)
        
        self.transform = transforms.Compose([
                transforms.ToTensor(),
        ])

    def __len__(self):
        #print('val Len: ', len(self.validation_data))
        return 62#len(self.validation_data)

    def Random_augmentation(self, image, mask):
        rnum = np.random.randint(0, 4)
        
        if rnum == 0:
            image = np.flip(image, 1)
            mask = np.flip(mask, 1)
        if rnum == 1:
            image = np.flipud(image)
            mask =  np.flipud(mask)
        if rnum == 2:
            image = np.flip(image, 1)
            mask = np.flip(mask, 1)
            image = np.flipud(image)
            mask =  np.flipud(mask)
        
        return image, mask

    def __getitem__(self, index):
        image_path, mask_path = self.validation_data[index]
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
         
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation = cv2.INTER_NEAREST)
        image = image[:, :, 0]
        mask = mask[:, :, 0]
        #print('before mask shape: ', mask.shape, ' ', np.unique(mask))

        imagec = image.copy()
        maskc = mask.copy()
        
        maskc[maskc == 255] = 1
        imagec = imagec/np.max(imagec)

        imagec = self.transform(imagec)
        #print('image shape: ', image.shape)
        #print('mask shape: ', maskc.shape, ' ', np.unique(maskc))
        #image = torch.Tensor(image)
        maskc = torch.Tensor(maskc)
        maskc = maskc.long()
        return imagec, maskc






