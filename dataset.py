# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
import os
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import numpy as np

class PredicDataset(Dataset):
    def __init__(self,root,transforms_=None,mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.real_stained_path=sorted((glob.glob(os.path.join(root,mode,'stained')+'/*.png')))
        file_path=os.path.join(root,mode)+'/label.txt'
        self.predict= self.preprocess(file_path)
        # print(self.predict)

    def preprocess(self,file_path):
        lines = [line.rstrip() for line in open(file_path, 'r')]
        file_type=lines[0].split()
        lines=lines[1:]
        values_vec={}
        for i,line in enumerate(lines):
            split=line.split()
            # print(split)
            file_name=split[0]
            values=split[1:]
            values_vec[file_name+'.png']=values
        return values_vec

    def __getitem__(self, item):
        # print(self.real_stained_path[item])
        stained_image=Image.open(self.real_stained_path[item])
        file_sqe=self.real_stained_path[item][27:]
        attr=self.predict[file_sqe]
        stained_item=self.transform(stained_image)
        attr_items=torch.from_numpy(np.array(attr,dtype=np.float32))

        # attr_item=torch.cat(attr)

        return stained_item,attr_items,file_sqe
    def __len__(self):
        return len(self.real_stained_path)

predict_transforms=[
    # transforms.Resize(int(256*1.68)),
    # transforms.RandomCrop((256,256)),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
]

# print(z)


class UseDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.unstain_file = sorted(glob.glob(os.path.join(root, 'input') + '/*.*'))

    def __getitem__(self, index):
        unstain_image = Image.open(self.unstain_file[index])
        unstain_item = self.transform(unstain_image)
        return unstain_item

    def __len__(self):
        return len(self.unstain_file)


class GenerateDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_stained = sorted(glob.glob(os.path.join(root, mode, 'stained') + '/*.*'))
        self.files_unstained = sorted(glob.glob(os.path.join(root, mode, 'unstained') + '/*.*'))

    def __getitem__(self, index):
        image_stained = Image.open(self.files_stained[index])
        if self.unaligned:
            image_unstained = Image.open(
                self.files_unstained[random.randint(0, len(self.files_unstained) - 1)]).convert('RGB')
        else:
            image_unstained = Image.open(self.files_unstained[index]).convert('RGB')

        sample = {'unstained': image_unstained, 'stained': image_stained}
        item_unstained = self.transform(sample['unstained'])
        item_stained = self.transform(sample['stained'])

        return {'unstained': item_unstained, 'stained': item_stained}

    def __len__(self):
        return len(self.files_stained)

class StainedDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_stained = sorted(glob.glob(os.path.join(root, mode, 'stained') + '/*.*'))
        self.files_unstained = sorted(glob.glob(os.path.join(root, mode, 'unstained') + '/*.*'))

    def __getitem__(self, index):
        image_stained = Image.open(self.files_stained[index])
        if self.unaligned:
            image_unstained = Image.open(
                self.files_unstained[random.randint(0, len(self.files_unstained) - 1)]).convert('RGB')
        else:
            image_unstained = Image.open(self.files_unstained[index]).convert('RGB')

        sample = {'unstained': image_unstained, 'stained': image_stained}
        resized_sample = Resize(int(1.12 * 256))(sample)
        crop_sample = RandomCrop((256, 256))(resized_sample)
        flip_sample = RandomFlip()(crop_sample)
        item_unstained = self.transform(flip_sample['unstained'])
        item_stained = self.transform(flip_sample['stained'])

        return {'unstained': item_unstained, 'stained': item_stained}

    def __len__(self):
        return len(self.files_stained)


class EvalDataset(Dataset):
    def __init__(self, root, mode='train', transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files_true = sorted(glob.glob(os.path.join(root, mode, 'true') + '/*.*'))
        self.files_generate = sorted(glob.glob(os.path.join(root, mode, 'generate') + '/*.*'))
        self.files_resgenerate = sorted(glob.glob(os.path.join(root, mode, 'resgenerate') + '/*.*'))

    def __getitem__(self, index):
        image_true = Image.open(self.files_true[index])
        image_generate = Image.open(self.files_generate[index])
        #        image_resgenerate=Image.open(self.files_resgenerate[index])
        item_true = self.transform(image_true)
        item_generate = self.transform(image_generate)
        # item_resgenerate=self.transform(image_resgenerate)
        return {'true': item_true, 'generate': item_generate}

    def __len__(self):
        return len(self.files_true)


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        unstained_image = sample['unstained']
        stained_image = sample['stained']
        w, h = unstained_image.size
        new_w, new_h = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        crop_unstained_image = unstained_image.crop((left, top, left + new_w, top + new_h))
        crop_stained_image = stained_image.crop((left, top, left + new_w, top + new_h))

        return {'unstained': crop_unstained_image, 'stained': crop_stained_image}


class Resize(object):
    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, sample):
        unstained_image = sample['unstained']
        stained_image = sample['stained']
        resized_unstained_image = unstained_image.resize(self.size, Image.BICUBIC)
        resized_stained_image = stained_image.resize(self.size, Image.BICUBIC)
        return {'unstained': resized_unstained_image, 'stained': resized_stained_image}


class RandomFlip(object):
    def __call__(self, sample):
        unstained_image = sample['unstained']
        stained_image = sample['stained']
        if torch.rand(1).item() > 0.5:
            flip_unstained_image = unstained_image.transpose(Image.FLIP_LEFT_RIGHT)
            flip_stained_image = stained_image.transpose(Image.FLIP_LEFT_RIGHT)
            return {'unstained': flip_unstained_image, 'stained': flip_stained_image}
        else:
            return {'unstained': unstained_image, 'stained': stained_image}
