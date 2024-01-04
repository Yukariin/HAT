from os import listdir
from os.path import join
import io
import random
import sqlite3

import cv2
import numpy as np
import imageio
from PIL import Image
import pickle
import torch
import torch.utils.data as data
from torchvision.transforms import RandomCrop, Resize
from torchvision.transforms import functional as TF


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def jpeg_noise_np(img_np, noise_level):
    if noise_level == 0:
        noise_level = [0, 5]
    elif noise_level == 1:
        noise_level = [5, 25]
    elif noise_level == 2:
        noise_level = [25, 50]
    else:
        raise KeyError("Noise level should be either 0, 1, 2")

    compression_level = 100 - np.random.randint(*noise_level)
    enc_img = cv2.imencode('.jpg', img_np, [int(cv2.IMWRITE_JPEG_QUALITY), compression_level])[1]

    out_img = cv2.imdecode(np.frombuffer(enc_img, np.uint8), -1)

    return out_img


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
    def __init__(self, image_dir,
                 patch_size=48, scale_factor=2,
                 interpolation=None,
                 hflip=True, rotate=True,
                 noise=None):
        super().__init__()

        self.samples = [join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x)]
        self.hflip = hflip
        self.rotate = rotate
        self.interpolation = interpolation
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.nl = noise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        hr = imageio.imread(sample_path, pilmode="RGB")
        h, w, _ = hr.shape
        sf = self.scale_factor

        if self.interpolation is None:
            # modes = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
            # probs = [0.10, 0.70, 0.10, 0.10]
            # inter = np.random.choice(modes, p=probs)
            modes = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
            inter = random.choice(modes)
        else:
            inter = self.interpolation
        lr = cv2.resize(hr, (w//sf, h//sf), interpolation=inter)

        if self.nl is not None:
            lr = jpeg_noise_np(lr, self.nl)
        
        def _random_crop(lr, hr, width, height):
            assert lr.shape[0] >= height
            assert lr.shape[1] >= width

            x = random.randint(0, lr.shape[1] - width)
            y = random.randint(0, lr.shape[0] - height)
            x_hr, y_hr = int(x * sf), int(y * sf)
            w_hr, h_hr = int(width * sf), int(height * sf)
            lr_patch = lr[y:y+height, x:x+width]
            hr_patch = hr[y_hr:y_hr+h_hr, x_hr:x_hr+w_hr]

            return lr_patch, hr_patch

        lr, hr = _random_crop(lr, hr, self.patch_size, self.patch_size)
        lr, hr = TF.to_tensor(lr), TF.to_tensor(hr)

        if self.hflip and random.random() < 0.5:
            lr = torch.flip(lr, [2])
            hr = torch.flip(hr, [2])
        if self.rotate and random.random() < 0.5:
            # vflip
            lr = torch.flip(lr, [1])
            hr = torch.flip(hr, [1])
        if self.rotate and random.random() < 0.5:
            # rot90
            lr = torch.transpose(lr, 1, 2)
            hr = torch.transpose(hr, 1, 2)

        return lr, hr


class II_Dataset(DatasetFromFolder):
    def __init__(self, image_dir, patch_size=48, scale_factor=2, interpolation=None,
                 hflip=True, rotate=True, data_partion=0.7, noise=None, val=False):
        super().__init__(image_dir, patch_size, scale_factor, interpolation, hflip, rotate)
        
        self.data_partion = data_partion
        self.nl = noise
        if val:
            self.val_index = 1000
        else:
            self.val_index = None
    
    def __getitem__(self, index):
        sample_path = self.samples[index]
        hr = imageio.imread(sample_path, pilmode="RGB")
        h, w, _ = hr.shape
        sf = self.scale_factor
        
        if self.interpolation is None:
            # modes = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
            # probs = [0.10, 0.70, 0.10, 0.10]
            # inter = np.random.choice(modes, p=probs)
            modes = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
            inter = random.choice(modes)
        else:
            inter = self.interpolation
        lr = cv2.resize(hr, (w//sf, h//sf), interpolation=inter)

        if self.nl is not None:
            lr = jpeg_noise_np(lr, self.nl)
        
        f_ii = sample_path.replace('.png', f'_ii_list_p{self.patch_size}.pt')
        with open(f_ii, 'rb') as _f:
            ii_index = pickle.load(_f)
        
        def _get_patch(lr, hr, ii_index, data_partion=0.7, patch_size=48, scale=2):
            n_patch = int(ii_index.shape[0] * data_partion)
            
            tp = patch_size * scale
            ip = tp // scale
            
            if n_patch == 0:
                index = 0
            elif n_patch > 0:    # positive order
                index = random.randrange(0, n_patch)
            else: # n_patch < 0: # reverse order
                index = random.randrange(n_patch, 0)
            if self.val_index is not None:
                index = self.val_index
            iy = int(ii_index[index][0])
            ix = int(ii_index[index][1])
            
            ty, tx = iy * scale, ix * scale
            
            return lr[iy:iy+ip, ix:ix+ip, :], hr[ty:ty+tp, tx:tx+tp, :]

        
        lr, hr = _get_patch(lr, hr, ii_index, self.data_partion, self.patch_size, self.scale_factor)
        lr, hr = TF.to_tensor(lr), TF.to_tensor(hr)

        if self.hflip and random.random() < 0.5:
            lr = torch.flip(lr, [2])
            hr = torch.flip(hr, [2])
        if self.rotate and random.random() < 0.5:
            # vflip
            lr = torch.flip(lr, [1])
            hr = torch.flip(hr, [1])
        if self.rotate and random.random() < 0.5:
            # rot90
            lr = torch.transpose(lr, 1, 2)
            hr = torch.transpose(hr, 1, 2)

        return lr, hr


class SQLDataset(data.Dataset):
    def __init__(self, db_file, db_table='images', lr_col='lr_img', hr_col='hr_img',
                 hflip=True, rotate=True):
        super().__init__()

        self.db_file = db_file
        self.db_table = db_table
        self.lr_col = lr_col
        self.hr_col = hr_col
        self.hflip = hflip
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
        if self.rotate and random.random() < 0.5:
            # vflip
            lr = torch.flip(lr, [1])
            hr = torch.flip(hr, [1])
        if self.rotate and random.random() < 0.5:
            # rot90
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
