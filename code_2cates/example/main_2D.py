import argparse
import os
import cv2
import sys
import json
import random
import traceback

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from openslide import OpenSlide

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from infer.predictor import ClsModel, ClsPredictor2DSigHead
except Exception:
    raise

def inference(predictor:ClsPredictor2DSigHead, hu_volume):
    pred_array = predictor.forward(hu_volume)
    return pred_array

def load_nii(nii_path):
    tmp_img = sitk.ReadImage(nii_path)
    spacing = tmp_img.GetSpacing()
    spacing = spacing[::-1]
    data_np = sitk.GetArrayFromImage(tmp_img)
    return data_np, tmp_img, spacing

def find_valid_region(mask, low_margin=[0,0,0], up_margin=[0,0,0]):
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

def flat(ll):
    out = []
    for l in ll:
        if isinstance(l, list):
            out.extend(l)
        else:
            out.append(l)
    return out

def split2batch(patches, batch_size):
    l = len(patches)
    out = []
    if l <= batch_size:
        short = batch_size - l
        b_patches = patches + random.choices(patches, k=short)
        out.append(b_patches)
    else:
        batch_num = int(np.ceil(l / batch_size))
        for i in range(batch_num):
            if len(patches) == 0:
                continue
            if len(patches) < batch_size:
                short = batch_size - len(patches)
                b_patches = patches + random.choices(patches, k=short)
                patches = []
            else:
                b_patches = random.sample(patches, batch_size)
                # patches = sorted(list(set(patches) - set(b_patches)))
                patches = [p for p in patches if p not in b_patches]
            out.append(b_patches)
    return out

def get_pids(lst_path):
    pids = []
    with open(lst_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')[0]
            pids.append(str(line))
    return pids

def main(input_path, output_path, gpu, args):
    
    model_path = args.model_path
    os.makedirs(output_path, exist_ok=True)
    model_files = [m for m in sorted(os.listdir(model_path)) if m.endswith('.pth')]
    for count, model_file in enumerate(model_files):
        cur_out_path = os.path.join(output_path, model_file.split('.p')[0])
        os.makedirs(cur_out_path, exist_ok=True)
        model_cls = ClsModel(
            model_f=os.path.join(model_path, model_file),
            network_f=args.network_file,
        )
        predictor_cls_2d = ClsPredictor2DSigHead(
            gpu = gpu,
            model = model_cls,
        )

        patient_pred_path = os.path.join(cur_out_path, 'patient_pred')
        
        os.makedirs(patient_pred_path, exist_ok=True)
        prefix = os.path.basename(os.path.dirname(output_path)).split('_')[0]

        labels_df = pd.read_csv('/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/data/labels.csv', encoding='utf8')
        labels_dict = dict()
        for i in range(labels_df.shape[0]):
            labels_dict[str(labels_df.iloc[i, 0])] = labels_df.iloc[i,-1]

        # pids = get_pids('/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/data_split/all_situ_pids.lst')
        # pids = get_pids('/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/data_split/all_lymph_pids.lst')
        pids = get_pids('/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/data_split/all_coexist_pids.lst')
        # pids = [str(p) for p in sorted(os.listdir(input_path)) if p.startswith('13')]
        for pid in tqdm(pids):
            print(pid)
            print('data load ......')
            all_flds = sorted(os.listdir(os.path.join(input_path, pid)))
            all_flds = [f for f in all_flds if os.path.isdir(os.path.join(input_path, pid, f))]
            label = labels_dict[pid]

            # for situ
            situ_flds = [f for f in all_flds if (f.startswith('A'))]
            # for lymph
            lymph_flds = [f for f in all_flds if (not f.startswith('A'))]
            # for coexist
            coexist_flds = [f for f in all_flds]
            if prefix == 'situ':
                valid_flds = situ_flds
            elif prefix == 'lymph':
                valid_flds = lymph_flds
            elif prefix == 'coexist':
                valid_flds = coexist_flds
            else:
                raise KeyError(f"wrong prefix: {prefix}")

            print(valid_flds)

            pid_patch_2D_probs = []
            if len(valid_flds) > 0:
                for folder in valid_flds:
                    jpgs = sorted(os.listdir(os.path.join(input_path, pid, folder)))
                    if len(jpgs) == 0:
                        continue
                    batches = split2batch(jpgs, batch_size)
                    for j, b_patches in enumerate(batches):
                        vol = []
                        patch_names = []
                        for p in b_patches:
                            img = cv2.imread(os.path.join(input_path, pid, folder, p))
                            img = np.array(img)
                            if len(img.shape) < 3:
                                continue
                            # convert colors
                            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                            # resize
                            if (np.array(img.shape[:2]) != np.array(patch_size[1:])).any():
                                img = cv2.resize(img, dsize=(patch_size[0], patch_size[1]),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
                            
                            # patch_test = './test'
                            # save_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            # os.makedirs(patch_test, exist_ok=True)
                            # cv2.imwrite('./test/'+pid+'_'+wsi_name+'_'+str(location[0])+'_'+str(location[1])+'_'+str(target_level)+'.jpg', img)

                            vol.append(img)
                            patch_names.append(p[:-4])

                        probs = inference(predictor_cls_2d, vol)
                        # patch level probability, size: batch_size
                        assert len(patch_names) == len(probs)
                        patch_2D_probs = [[patch_names[i], label, probs[i]] for i in range(len(patch_names))]
                        pid_patch_2D_probs.extend(patch_2D_probs)
                        print(f"{pid}-{folder}-batch{j} inference completed!")

            pd.DataFrame(
                pid_patch_2D_probs, columns=['patch index', 'label', 'prob']
            ).to_csv(os.path.join(patient_pred_path, pid+'_2d_prob.csv'), index=None, encoding='utf8')

predata_path = '/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/predata_level2_margin_tdot9_sttd'
patch_size = [256, 256] 
batch_size = 300
def parse_args():
    parser = argparse.ArgumentParser(description='Test for pathology')

    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--input_path', default=predata_path, type=str)
    parser.add_argument('--output_path', default='./data_pathology_2D/situ_level2/', type=str)

    parser.add_argument('--target_level', default=0, type=int)
    parser.add_argument('--model_path', default='./data_pathology_2D/situ_level2/', type=str,)
    parser.add_argument('--network_file', default='./data_pathology_2D/config.py', type=str,)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    fine_seg = False
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        gpu=args.gpu,
        args=args,
    )
