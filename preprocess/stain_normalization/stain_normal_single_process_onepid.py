import os
import cv2
import time
import spams
import datetime
import traceback
import numpy as np
import random
from vahadane  import VahadaneNormalizer
from macenko import MacenkoNormalizer
from multiprocessing import Pool, Process

from utils import misc_utils as mu

def standard_transfrom(standard_img,method = 'V'):
    if method == 'V':
        stain_method = VahadaneNormalizer()
        stain_method.fit(standard_img)
    else:
        stain_method = MacenkoNormalizer()
        stain_method.fit(standard_img)
    return stain_method

def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # p = np.percentile(img, 90)
    # img = np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)
    return img

if __name__ == "__main__":
    sttd_path = '/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/predata_level2_margin_tdot9/refer_bgr_img.jpg'
    sttd_img = read_image(sttd_path)
    stain_method = standard_transfrom(sttd_img)

    patch_dir = "/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/predata_level2_margin_tdot9"
    out_dir = patch_dir
    # pids = os.listdir(patch_dir)
    # pids = sorted([str(p) for p in pids if p.startswith('13')])
    # pids = ['137158']
    # pids = ['133150']
    # pids = ['135974']
    pids = ['135904']
    for p in pids:
        pid_info = []
        all_files = sorted(os.listdir(os.path.join(patch_dir, p)))
        # patch_folders = [f for f in all_files if os.path.isdir(os.path.join(patch_dir, p, f))]
        # patch_folders = ['DE133150', 'FG133150', 'IH133150', 'BC137118']
        # patch_folders = ['DE135974', 'FG135974', 'HI135974', 'JK135974']
        patch_folders = ['LM135904', 'JK135904']
        for patch_folder in patch_folders:
            start_time = time.time()
            cur_out_dir = os.path.join(out_dir, p, patch_folder+'_sttd')
            os.makedirs(cur_out_dir, exist_ok=True) 
            jpgs = sorted(os.listdir(os.path.join(patch_dir, p, patch_folder)))
            if len(jpgs) == 0:
                continue
            elif len(jpgs) > 1000:
                jpgs = random.sample(jpgs, k=1000)
            for jpg in jpgs:
                print(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'), '  ',jpg)
                if os.path.exists(os.path.join(cur_out_dir, jpg)):
                    continue
                jpg_path = os.path.join(patch_dir, p, patch_folder, jpg)
                bgr_img = cv2.imread(jpg_path)
                I = mu.standardize_brightness(bgr_img)
                test_mask = mu.notwhite_mask(I, thresh=0.8).reshape((-1,))
                if test_mask.sum() < 10:
                    continue
                try:
                    bgr_img2 = stain_method.transform(bgr_img)
                    cv2.imwrite(os.path.join(cur_out_dir, jpg), bgr_img2)
                except:
                    traceback.print_exc()
                    print(f"{jpg} in {patch_folder} can't be stained!")
                    continue
            end_time = time.time()
            print(f"============{p}: {patch_folder} completed and cost time: {int(end_time-start_time)} s!============")

                

    

