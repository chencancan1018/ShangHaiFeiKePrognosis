"""data loader."""

import os
import json
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
from openslide import OpenSlide
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
    #    albumentations.Normalize(always_apply=True, p=1.0),
       ToTensorV2(),
    ])


def flat(ll):
    out = []
    for l in ll:
        if isinstance(l, list):
            out.extend(l)
        else:
            out.append(l)
    return out

# @DATASETS.register_module
class Cls2DSampleDataset(DefaultSampleDataset):
    def __init__(
            self,
            dst_list_file,
            data_root,
            patch_size,
            target_level,
            sample_frequent,
            pipelines,
    ):
        self._sample_frequent = sample_frequent
        self._patch_size = patch_size
        self.target_level = target_level
        self._data_file_list = self._load_file_list(data_root, dst_list_file)
        self.draw_idx = 1
        self.transforms = transforms
        if len(pipelines) == 0:
            self.pipeline = None
        else:
            self.pipeline = build_pipelines(pipelines)

    def _load_file_list(self, data_root, dst_list_file):
        data_file_list = []
        with open(dst_list_file, "r") as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                file_name = line[0]
                file_name = os.path.join(data_root, str(file_name))
                if not os.path.exists(file_name):
                    print(f"{line} not exist")
                    continue
                data_file_list.append(file_name)
        random.shuffle(data_file_list)
        assert len(data_file_list) != 0, "has no avilable file in dst_list_file"
        return data_file_list

    def _load_source_data(self, file_name):
        json_files = os.listdir(file_name)
        # json_files = [f for f in json_files if f.startswith('A')] 
        json_files = [f for f in json_files if not f.startswith('A')]
        if len(json_files) > 1:
            json_file = random.sample(json_files, k=1)[0]
        elif len(json_files) == 1:
            json_file = json_files[0]
        else:
            raise KeyError(f"{file_name} not exist valid json file!")
        with open(os.path.join(file_name, json_file), 'r') as f:
            pid_info_dict = json.load(f)
        result = {}
        with torch.no_grad():
            label = pid_info_dict['label']
            label = torch.from_numpy(np.array([label]))
            patches = pid_info_dict['patches']
            patches = flat(patches)

            p = random.sample(patches, 1)[0]
            wsi_path = p['wsi_path']
            slide = OpenSlide(wsi_path)
            downsample = pow(2, int(p['level'])-self.target_level)
            downsample2level0 = pow(2, int(p['level']))
            location = (int(p['start_x'] * downsample2level0), int(p['start_y'] * downsample2level0))
            size = (int(p['w'] * downsample), int(p['h'] * downsample))
            
            # OpenSlide read data
            img = slide.read_region(location, self.target_level, size)
            # convert colors
            img = np.array(img)
            rgb_arr = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            # resize
            if (np.array(rgb_arr.shape[:2]) != np.array(self._patch_size[1:])).any():
                rgb_arr = cv2.resize(rgb_arr, dsize=(self._patch_size[1], self._patch_size[2]),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
            # augmentation
            img = self.transforms(image=rgb_arr)['image']
            # normalize 
            img = img / 255 # to [0,1]
            img = (img - 0.5) / 0.5 # to [-1, 1]

            # tensor
            assert img.shape[0] == 3
            assert img.shape[1] == self._patch_size[0]
            assert img.shape[2] == self._patch_size[1]

            result['vol'] = img.detach()
            result['label'] = label.detach()
        del pid_info_dict
        del slide
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


    