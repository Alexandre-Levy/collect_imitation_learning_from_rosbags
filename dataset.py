from pathlib import Path

import numpy as np
import torch
import imgaug.augmenters as iaa
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from numpy import nan
import noise
import os

from converter import Converter, PIXELS_PER_WORLD
import common
import pathlib
import cv2
import json

# Reproducibility.
np.random.seed(0)
torch.manual_seed(0)

# Data has frame skip of 5.
GAP = 4 # 1
STEPS = 29 # 4
N_CLASSES = len(common.COLOR)

class CarlaDataset(Dataset):
    def __init__(self, dataset_dir, transform=transforms.ToTensor(), noise_threshold_percentage=20, noise_threshold_percentage_back=20):
        dataset_dir = Path(dataset_dir)
        f_img = open(dataset_dir / 'measurements_quadratic.json')
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.frames = list()
        self.measurements = json.load(f_img)
        self.converter = Converter()

        print(dataset_dir)

        for image_path in sorted((dataset_dir / 'img').glob('*.png')):
            frame = str(image_path.stem)

            assert (dataset_dir / 'img' / ('%s.png' % frame)).exists()

            # assert int(frame) < len(self.measurements)

            self.frames.append(frame)

        assert len(self.frames) > 0, '%s has 0 frames.' % dataset_dir

    def __len__(self):
        return len(self.frames) - GAP * STEPS

    def __getitem__(self, value):
        path = self.dataset_dir
        frame = self.frames[value]
        meta = '%s %s' % (path.stem, frame)

        shape = (256,256)
        scale = 100.0 #100.0
        octaves = 10 #10
        persistence = 0.5 #0.5
        lacunarity = 5.0 #2.0
        nb_patches = 16
        nb_patches_noisy = 3
        nb_rand = 100000

        rgb = Image.open(path / 'img' / ('%s.png' % frame))
        # print(rgb.size)
        rgb = transforms.functional.to_tensor(rgb)

        # topdown = Image.open(path / 'topdown_partial_depth_segmented_prediction' / ('%s.png' % frame))
        # topdown_im = topdown.crop((128, 0, 128 + 256, 256))
        # topdown = np.array(topdown_im)
        
        # topdown = preprocess_semantic(topdown)
        u = np.float32([self.measurements[str(value)]['x'], self.measurements[str(value)]['y']])
        theta = self.measurements[str(value)]['theta']
        # print(theta)
        if np.isnan(theta):
            theta = 0.0
        theta = theta * np.pi / 180 # - np.pi / 4 # + theta_offset
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        points = list()

        for skip in range(1, STEPS+1):# (1,STEPS+1):
            j = value + GAP * skip
            v = np.array([self.measurements[str(j)]['x'], self.measurements[str(j)]['y']])
            target = R.T.dot(v - u)
            # print(target)
            target *= PIXELS_PER_WORLD
            target += [128, 256]
            points.append(target)

        points = torch.FloatTensor(np.vstack(points).astype(np.float))
        points = torch.clamp(points, 0, 256)
        points = (points / 256) * 2 - 1

        return rgb, points#, target, #actions, meta

class ToHeatmap(torch.nn.Module):
    def __init__(self, radius=5):
        super().__init__()

        bounds = torch.arange(-radius, radius+1, 1.0)
        y, x = torch.meshgrid(bounds, bounds)
        kernel = (-(x ** 2 + y ** 2) / (2 * radius ** 2)).exp()
        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())

        self.r = radius
        self.register_buffer('kernel', kernel)

    def forward(self, points, img):
        n, _, h, w = img.shape
        heatmap = torch.zeros((n, h, w)).type_as(img)

        for i in range(n):
            output = heatmap[i]

            cx, cy = points[i].round().long()
            cx = torch.clamp(cx, 0, w-1)
            cy = torch.clamp(cy, 0, h-1)

            left = min(cx, self.r)
            right = min(w - 1 - cx, self.r)
            bot = min(cy, self.r)
            top = min(h - 1 - cy, self.r)

            output_crop = output[cy-bot:cy+top+1, cx-left:cx+right+1]
            kernel_crop = self.kernel[self.r-bot:self.r+top+1, self.r-left:self.r+right+1]
            output_crop[...] = kernel_crop

        return heatmap

if __name__ == '__main__':
    import sys
    import cv2
    from PIL import ImageDraw

    data = CarlaDataset(sys.argv[1])
    converter = Converter()
    to_heatmap = ToHeatmap()
    pathlib.Path(f'{sys.argv[1]}/results_projection_quadratic').mkdir(parents=True, exist_ok=True)
    parameters_file = open(sys.argv[1])
    parameters = json.load(parameters_file)
    for i in range(len(data)):
        rgb, points = data[i]
        
        points_unnormalized = (points + 1) / 2 * 256
        points_cam = converter(points_unnormalized)


        _rgb = (rgb.cpu() * 255).byte().numpy().transpose(1, 2, 0)[:, :, :3]
        _rgb = Image.fromarray(_rgb)
        _rgb = _rgb.resize((406,256))

        _topdown = Image.new(mode="RGB", size=(256,256))


        _draw_map = ImageDraw.Draw(_topdown)
        _draw_rgb = ImageDraw.Draw(_rgb)

        for x, y in points_unnormalized:
            _draw_map.ellipse((x-1, y-1, x+1,y+1), (255, 0, 0))

        for x, y in points_cam:
            _draw_rgb.ellipse((x-2, y-2, x+2,y+2), (255, 0, 0))

        # _topdown.thumbnail(_rgb.size)
        # cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
        # cv2.imshow('debug', cv2.cvtColor(np.hstack((_rgb, _topdown)), cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)
        cv2.imwrite((f'{sys.argv[1]}/results_projection_quadratic/%04d.png' % i), cv2.cvtColor(np.hstack((_rgb, _topdown)), cv2.COLOR_BGR2RGB))
