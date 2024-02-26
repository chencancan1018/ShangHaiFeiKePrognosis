"""data loader."""

import os
import random
import traceback

import numpy as np
import torch
from starship.umtf.common import build_pipelines
from starship.umtf.common.dataset import DATASETS
from starship.umtf.service.component import CustomDataset, DefaultSampleDataset
from scipy.ndimage import zoom

import cv2
import albumentations
from albumentations.pytorch import ToTensorV2
"""
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
Augment an image
"""
transforms = albumentations.Compose([
       albumentations.VerticalFlip(p=0.5),
       albumentations.HorizontalFlip(p=0.5),
       albumentations.HueSaturationValue(hue_shift_limit=20,sat_shift_limit=30,val_shift_limit=20,p=0.5),
       albumentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
       albumentations.Rotate(border_mode=cv2.BORDER_CONSTANT,value=0,p=0.5),
       albumentations.RandomSizedCrop((256,256), 256,256),
       ToTensorV2(),
    ])

# @DATASETS.register_module
class Cls25DSampleDataset(DefaultSampleDataset):
    def __init__(
            self,
            dst_list_file,
            data_root,
            patch_size,
            sample_frequent,
            win_level,
            win_width,
            pipelines,
    ):
        self._sample_frequent = sample_frequent
        self._patch_size = patch_size
        self._data_file_list = self._load_file_list(data_root, dst_list_file)
        self.draw_idx = 1
        self._win_level = win_level
        self._win_width = win_width
        if len(pipelines) == 0:
            self.pipeline = None
        else:
            self.pipeline = build_pipelines(pipelines)

    def _load_file_list(self, data_root, dst_list_file):
        data_file_list = []
        with open(dst_list_file, "r") as f:
            for line in f:
                line = line.strip().split(' ')
                file_name = line[0]
                file_name = os.path.join(data_root, file_name)
                if not os.path.exists(file_name):
                    print(f"{line} not exist")
                    continue
                data_file_list.append(file_name)
        random.shuffle(data_file_list)
        assert len(data_file_list) != 0, "has no avilable file in dst_list_file"
        return data_file_list

    def find_valid_region(self, mask, low_margin=[0,0,0], up_margin=[0,0,0]):
        nonzero_points = np.argwhere((mask > 0))
        if len(nonzero_points) == 0:
            return None, None
        else:
            v_min = np.min(nonzero_points, axis=0)
            v_max = np.max(nonzero_points, axis=0)
            assert len(v_min) == len(low_margin), f'the length of margin is not equal the mask dims {len(v_min)}!'
            for idx in range(len(v_min)):
                v_min[idx] = max(0, v_min[idx] - low_margin[idx])
                v_max[idx] = min(mask.shape[idx], v_max[idx] + up_margin[idx])
            return v_min, v_max

    def _load_source_data(self, file_name):
        # print(file_name)
        data = np.load(file_name.split('\n')[0])
        result = {}
        with torch.no_grad():
            vol = data["vol"]
            seg = data["seg"]
            label = data['label']

            # 肺区
            pmin, pmax = self.find_valid_region(seg.copy())
            vol = vol[pmin[0]:(pmax[0]+1), pmin[1]:(pmax[1]+1), pmin[2]:(pmax[2]+1)]
            seg = seg[pmin[0]:(pmax[0]+1), pmin[1]:(pmax[1]+1), pmin[2]:(pmax[2]+1)]

            # 加窗
            vol = [self.window_array(vol, wl, wd)[None] for wl, wd in zip(self._win_level, self._win_width)]
            vol = np.concatenate(vol, axis=0) # channel first
            seg = seg[None] # channel first
            vol_shape = np.array(vol.shape[1:])
            seg_shape = np.array(seg.shape[1:])

            # resize
            patch_size = np.array(self._patch_size)
            if np.any(vol_shape != patch_size):
                scale = np.array(np.array([vol.shape[0]] + list(patch_size)) / np.array(vol.shape))
                vol = zoom(vol, scale, order=1)
            
            if np.any(patch_size != seg_shape):
                scale = np.array(np.array([seg.shape[0]] + list(patch_size)) / np.array(seg.shape))
                seg = zoom(seg, scale, order=0)

            assert (np.array(vol.shape[1:]) == np.array(seg.shape[1:])).all()
            # numpy2tensor
            vol = torch.from_numpy(vol).float()
            seg = torch.from_numpy(seg).float()

            # augmentation
            if self.pipeline:
                vol, seg = self.pipeline(data=(vol, seg))
            
            slices = []
            slice_0 = torch.cat([vol[0, 0, :, :][None], vol[0, 0, :, :][None], vol[0, 1, :, :][None], seg[0, 0, :, :][None]], dim=0)
            slices.append(slice_0[None])
            for i in range(1, vol.shape[1]-1):
                slice_i = torch.cat([vol[0, i-1, :, :][None], vol[0, i, :, :][None], vol[0, i+1, :, :][None], seg[0, i, :, :][None]], dim=0)
                slices.append(slice_i[None])
            last_idx = vol.shape[1]-1
            slice_last = torch.cat([vol[0, last_idx-1, :, :][None], vol[0, last_idx, :, :][None], vol[0, last_idx, :, :][None], seg[0, last_idx, :, :][None]], dim=0)
            slices.append(slice_last[None])
            vol = torch.cat(slices, dim=0)
            assert vol.shape[0] == patch_size[0]
            assert vol.shape[1] == 4
            assert vol.shape[2] == patch_size[1]
            assert vol.shape[3] == patch_size[2]

            if label == '19':
                label = torch.from_numpy(np.array([0]))
            elif label == '21':
                label = torch.from_numpy(np.array([1]))
            elif label =='other':
                label = torch.from_numpy(np.array([2]))

            result['vol'] = vol.detach()
            result['label'] = label.detach()
        del data
        return result

    def _sample_source_data(self, idx, source_data_info):
        try:
            info, _source_data = source_data_info
            return _source_data
        except:
            traceback.print_exc()
            return None
    
    def sample_source_data(self, idx, source_data):
        sample = None
        if idx < self._sample_frequent:
            sample = self._sample_source_data(idx, source_data)
        return sample

    def __getitem__(self, idx):
        source_data = self._load_source_data(self._data_file_list[idx])
        return [None, source_data]

    @property
    def sampled_data_count(self):
        # TODO: sample后数据总数量
        return self.source_data_count * self._sample_frequent

    @property
    def source_data_count(self):
        # TODO: 原始数据总数量
        return len(self._data_file_list)

    def __len__(self):
        return self.source_data_count


    def window_array(self, vol, win_level, win_width):
        win = [
            win_level - win_width / 2,
            win_level + win_width / 2,
        ]
        vol = np.clip(vol, win[0], win[1])
        vol -= win[0]
        vol /= win_width
        return vol

    