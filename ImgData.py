from __future__ import print_function, division
import torch as t
import os
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# im2latex_train/test/val, im2formulas, png_dir
class ImgDataSet(Dataset):
    def __init__(self, latex_nth2imgName,
                 nth_latex, img_dir, transform=None):
        image2nth_dict = open(latex_nth2imgName, 'r')
        nth2latex_dict = open(nth_latex, 'r', encoding='UTF-8')
        self.nth2name={}
        self.name2nth={}
        self.nth2latex={}
        self.transform = transform
        for line in image2nth_dict:
            temp = line.split()
            self.nth2name[temp[0]] = temp[1]
            self.name2nth[temp[1]] = temp[0]
        i = 0
        for line in nth2latex_dict:
            self.nth2latex[str(i)] = line
            i += 1
        self.img_dir = img_dir

    def __len__(self):
        return len(self.nth2name)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.nth2name[str(idx)])
        image = io.imread(img_name)
        sample = {'image': image, 'latex':self.nth2latex[str(idx)]}

        if self.transform:
            sample = self.transform(sample)

        return sample

class SmallImgDataSet(Dataset):
    def __init__(self, max_latex_length, latex_nth2img,
                 latex_formulas, img_dir, transform=None):
        img_nth = open(latex_nth2img, 'r') # no. --- name
        formulas = open(latex_formulas, 'r', encoding='UTF-8') # formulas
        self.nth2name = {}
        self.name2nth = {}
        self.nth2latex = {}
        self.ith2nth = {}
        self.transform = transform
        for line in img_nth:
            temp = line.split()
            self.nth2name[temp[0]] = temp[1]
            self.name2nth[temp[1]] = temp[0]
        i = 0
        for n, line in enumerate(formulas):
            self.nth2latex[str(n)] = line
            if len(line)<=max_latex_length:
                self.ith2nth[i] = n
                i += 1
            n += 1
        self.img_dir = img_dir

    def __len__(self):
        return len(self.ith2nth)

    def __getitem__(self, idx):
        nth = self.ith2nth[idx]
        img_name = self.nth2name[str(nth)] + '.png'
        img_name = os.path.join(self.img_dir, img_name)
        image = io.imread(img_name)
        sample = {'image': image, 'latex':self.nth2latex[str(nth)]}
        if self.transform:
            sample = self.transform(sample)
        return sample

class UniChannel(object):
    def __call__(self, sample):
        image, latex = sample['image'], sample['latex']
        image = image[:, :, 0]
        image = image[:, :, np.newaxis]
        return {'image':image, 'latex':latex}

class Rescale(object):
    def __init__(self, output_sz):
        assert isinstance(output_sz, (int, tuple))
        self.output_sz = output_sz

    def __call__(self, sample):
        image, latex = sample['image'], sample['latex']

        h, w = image.shape[:2]
        if isinstance(self.output_sz, int):
            if h > w:
                new_h, new_w = self.output_sz * h / w, self.output_sz
            else:
                new_h, new_w = self.output_sz, self.output_sz * w / h
        else:
            new_h, new_w = self.output_sz
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image,(new_h, new_w))
        return {'image':img, 'latex':latex}

class ToTensor(object):
    def __call__(self, sample):
        image, latex = sample['image'], sample['latex']
        image = image.transpose((2, 0, 1))
        return {'image':t.from_numpy(image), 'latex':latex}

class FormulaSym:
    def __init__(self):
        self.sym2index = {}
        self.sym2count = {}
        self.index2sym = {0:"SOS", 1:"EOS"}
        self.n_ch = 2

    def add_formula(self, formula):
        for ch in formula:
            self.add_ch(ch)

    def add_ch(self, ch):
        if ch not in self.sym2index:
            self.sym2index[ch] = self.n_ch
            self.sym2count[ch] = 1
            self.index2sym[self.n_ch] = ch
            self.n_ch += 1
        else:
            self.sym2count[ch] += 1

    def __len__(self):
        return self.n_ch

    def __getitem__(self, ch):
        return self.sym2index[ch]
