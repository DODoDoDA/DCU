import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch
import torch.nn as nn   
import random

imagenet_templates_smallest = [
    'a photo of a {}',
]

imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
    'an illustration of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'an illustration of a clean {}',
    'an illustration of a dirty {}',
    'a dark photo of the {}',
    'an illustration of my {}',
    'an illustration of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'an illustration of the {}',
    'a good photo of the {}',
    'an illustration of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'an illustration of the clean {}',
    'a rendition of a {}',
    'an illustration of a nice {}',
    'a good photo of a {}',
    'an illustration of the nice {}',
    'an illustration of the small {}',
    'an illustration of the weird {}',
    'an illustration of the large {}',
    'an illustration of a cool {}',
    'an illustration of a small {}',
    'a depiction of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a depiction of a clean {}',
    'a depiction of a dirty {}',
    'a dark photo of the {}',
    'a depiction of my {}',
    'a depiction of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a depiction of the {}',
    'a good photo of the {}',
    'a depiction of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a depiction of the clean {}',
    'a rendition of a {}',
    'a depiction of a nice {}',
    'a good photo of a {}',
    'a depiction of the nice {}',
    'a depiction of the small {}',
    'a depiction of the weird {}',
    'a depiction of the large {}',
    'a depiction of a cool {}',
    'a depiction of a small {}',
]



class GenDataset(Dataset):
    def __init__(self,
                 input_np_array,
                 class_ids,
                 size=256,
                 placeholder_token="*",
                 min_class_id=0,
                 ):
        self.img_array = input_np_array #imgs
        self.class_ids = class_ids #labels
        self.min_class_id = min_class_id
        self.placeholder_token = placeholder_token
        self.size = size # img resize to size * size
        self.interpolation = PIL.Image.BICUBIC
        self.flip = transforms.RandomHorizontalFlip(p=0.2)
    def __len__(self):
        return len(self.img_array)

    def __getitem__(self, i):
        example = {}
        image = Image.fromarray(self.img_array[i])
        class_id = self.class_ids[i]
        example["abs_class_id"] = class_id
        example["rel_class_id"] = class_id - self.min_class_id

        if not image.mode == "RGB":
            image = image.convert("RGB")
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        #'a photo of a *'
        text = random.choice(imagenet_templates_small).format(self.placeholder_token)
        example["caption"] = text
        
        return example
    
class GenDataset_Mnist(Dataset):
    def __init__(self, np_imgs, np_labels, num_classes, min_class_id=0):
        self.train_data = np_imgs #imgs
        self.train_targets = np_labels #labels
        self.n_classes = num_classes
        self.min_class_id = min_class_id
        self.transform = transforms.Compose([
                    transforms.Resize((32,32)),
                    transforms.ToTensor(),
                    lambda x: 2*(x - 0.5)
                ])
        self.target_transform = transforms.Compose([
            lambda x: torch.tensor([x]),
            lambda class_labels, n_classes=self.n_classes: nn.functional.one_hot(class_labels, n_classes).squeeze()
        ])
    def __getitem__(self,index):
        img = self.train_data[index]
        real_label = self.train_targets[index]
        label = self.train_targets[index] - self.min_class_id
        img = Image.fromarray(img)  
        if self.transform is not None: 
            img = self.transform(img) 
        if self.target_transform is not None: 
            label = self.target_transform(label) 
        return img, label, real_label
    
    def __len__(self):
        return len(self.train_data)  
