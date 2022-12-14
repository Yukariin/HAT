from os import listdir
from os.path import join
import io
import random
import sqlite3

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision.transforms import RandomCrop, Resize
from torchvision.transforms import functional as TF


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class DatasetFromList(data.Dataset):
    def __init__(self, list_path, patch_size=48, scale_factor=2, interpolation=Image.BICUBIC):
        super().__init__()

        self.samples = [x.rstrip('\n') for x in open(list_path) if is_image_file(x.rstrip('\n'))]
        self.cropper = RandomCrop(patch_size * scale_factor)
        self.resizer = Resize(patch_size, interpolation)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        img = Image.open(sample_path).convert('RGB')

        target = self.cropper(img)
        input = target.copy()
        input = self.resizer(input)

        return TF.to_tensor(input), TF.to_tensor(target)


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, patch_size=48, scale_factor=2, interpolation=None,
                 hflip=True, vflip=True, rotate=True):
        super().__init__()

        self.samples = [join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x)]
        self.cropper = RandomCrop(patch_size * scale_factor)
        self.resizer = Resize(patch_size, interpolation)
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate
        self.interpolation = interpolation
        self.patch_size = patch_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        img = Image.open(sample_path).convert('RGB')

        hr = self.cropper(img)

        if self.interpolation is None:
            interpolation = random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
        else:
            interpolation = self.interpolation
        lr = hr.resize((self.patch_size, self.patch_size), interpolation)

        lr, hr = TF.to_tensor(lr), TF.to_tensor(hr)

        if self.hflip and random.random() < 0.5:
            lr = torch.flip(lr, [2])
            hr = torch.flip(hr, [2])
        if self.vflip and random.random() < 0.5:
            lr = torch.flip(lr, [1])
            hr = torch.flip(hr, [1])
        if self.rotate and random.random() < 0.5:
            lr = torch.transpose(lr, 1, 2)
            hr = torch.transpose(hr, 1, 2)

        return lr, hr


class SQLDataset(data.Dataset):
    def __init__(self, db_file, db_table='images', lr_col='lr_img', hr_col='hr_img',
                 hflip=True, vflip=True, rotate=True):
        super().__init__()

        self.db_file = db_file
        self.db_table = db_table
        self.lr_col = lr_col
        self.hr_col = hr_col
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate
        self.total_images = self.get_num_rows()

    def get_num_rows(self):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT MAX(ROWID) FROM {self.db_table}')
            db_rows = cursor.fetchone()[0]

        return db_rows

    def __len__(self):
        return self.total_images

    def __getitem__(self, item):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT {self.lr_col}, {self.hr_col} FROM {self.db_table} WHERE ROWID={item+1}')
            lr, hr = cursor.fetchone()

        lr = Image.open(io.BytesIO(lr)).convert('RGB')
        hr = Image.open(io.BytesIO(hr)).convert('RGB')

        lr, hr = TF.to_tensor(lr), TF.to_tensor(hr)

        if self.hflip and random.random() < 0.5:
            lr = torch.flip(lr, [2])
            hr = torch.flip(hr, [2])
        if self.vflip and random.random() < 0.5:
            lr = torch.flip(lr, [1])
            hr = torch.flip(hr, [1])
        if self.rotate and random.random() < 0.5:
            lr = torch.transpose(lr, 1, 2)
            hr = torch.transpose(hr, 1, 2)

        return lr, hr


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0
