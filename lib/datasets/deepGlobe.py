from torch.utils.data import Dataset
import torch
from skimage import io
import os
import pandas as pd
import numpy as np
import matplotlib
import pdb


class DGRoad(Dataset):
    def __init__(self, mode='train', load_file='train_all_rgb-ps', cfg=None, transforms=None):
        self.mode = mode
        self.load_file = load_file
        self.cfg = cfg
        self.data_dir = cfg["data_dir"]
        self._img_index = self._load_image_set_index()
        self.transforms = transforms

    def __len__(self):
        return len(self._img_index)

    def __getitem__(self, idx):
        if not (self.mode == 'test'):
            strstr = self._img_index[idx]
            # print("====={}=====".format(strstr))
            sat_img_name = os.path.join(self.data_dir, "val_crops", "images","{}.png".format(strstr))

            sat_image = io.imread(sat_img_name)
            # print(sat_image.shape)
            map_img_name = os.path.join(self.data_dir, "val_crops", "gt","{}.png".format(strstr))
            map_image = io.imread(map_img_name)
            # print(map_image.shape)
            # map_image = map_image[:,:,0]
            map_image[map_image > 0] = 1
            img, mask, ori = self.transforms(sat_image, map_image)
            img_input = torch.Tensor(np.moveaxis(img, -1, 0).copy())
            mask_input = torch.Tensor(np.expand_dims(mask, 0).copy())
            return {'img': img_input, 'mask': mask_input, 'ori_img': ori}

        else:
            strstr = self._img_index[idx]
            # print("====={}=====".format(strstr))
            sat_img_name = os.path.join(self.data_dir, strstr.split(',')[0])
            print(sat_img_name.size())
            sat_image = io.imread(sat_img_name)
            map_img_name = os.path.join(self.data_dir, strstr.split(',')[1])
            map_image = io.imread(map_img_name)
            map_image[map_image > 0] = 1
            split_point = self.cfg['split_point']
            # img, mask, ori = self.transforms(sat_image, map_image)
            # img_input = torch.Tensor(np.moveaxis(img, -1, 0).copy())
            # mask_input = torch.Tensor(np.expand_dims(mask, 0).copy())

            img_inputs = []
            for i in range(len(split_point)):
                for j in range(len(split_point)):
                    x1 = int(split_point[i])
                    x2 = int(split_point[i] + 512)
                    y1 = int(split_point[j])
                    y2 = int(split_point[j] + 512)

                    crop_img = sat_image[x1:x2, y1:y2].copy()
                    crop_mask = map_image[x1:x2, y1:y2].copy()

                    img, mask, ori = self.transforms(crop_img, crop_mask)
                    img_input = torch.Tensor(np.moveaxis(img, -1, 0).copy())
                    mask_input = torch.Tensor(np.expand_dims(mask, 0).copy())
                    img_inputs.append({'img': img_input, 'mask': mask_input})
            return img_inputs
            # return {'img': img_input, 'mask': mask_input}

    def _load_image_set_index(self):
        image_set_file = os.path.join(self.data_dir, 'list', self.load_file + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index[::4]
