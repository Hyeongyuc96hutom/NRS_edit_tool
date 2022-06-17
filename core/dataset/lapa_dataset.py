import os
import random
import numpy as np
import torch
from glob import glob
from PIL import Image

from torch.utils.data import Dataset
from core.config.data_info import data_transforms


class LapaDataset(Dataset):
    def __init__(self, args, state='train'):
        super().__init__()

        self.args = args
        self.mode_hem = False
        self.ids = None
        self.x = None
        self.y = None

        if state == 'train':
            self.aug = data_transforms['train']
        elif state == 'val':
            self.aug = data_transforms['val']
        elif state == 'test':
            self.aug = data_transforms['test']


    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.mode_hem:
            pass
        else:
            pass

        return 0

    def change_mode(self, to_hem=False):
        self.mode_hem = to_hem

    def set_sample_ids(self, ids):
        self.ids = ids

    def load_data(self):
        if self.data_version == 'v1':
            self.load_v1()
        elif self.data_version == 'v2':
            self.load_v2()

    def load_v1(self):
        # TODO load dataset ver. 1
        pass

    def load_v2(self):
        # TODO load dataset ver. 2
        pass