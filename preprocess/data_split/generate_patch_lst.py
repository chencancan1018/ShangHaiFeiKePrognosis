import os
import os.path as osp
import random
import numpy as np
import pandas as pd

def get_pids(lst_path):
    pids = []
    with open(lst_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            pids.append(str(line))
    pids = sorted(pids)
    return pids

def lst_to_save(pids_lst, outpath):
    with open(outpath, 'w') as f:
        for p in pids_lst:
            f.writelines(p+'\n')

def folder_selected(prefix, dirs):
    if prefix == 'situ':
        dirs = [d for d in dirs if d.startswith('A')]
    elif prefix == 'lymph':
        dirs = [d for d in dirs if not d.startswith('A')]
    elif prefix == 'coexist':
        dirs = dirs
    else:
        raise KeyError(f"wrong prefix: {prefix}")
    return dirs

def gen_single_lst(prefix, input_dir, out_dir):
    train_pids = get_pids(prefix + '_train_pids.lst')
    test_pids = get_pids(prefix + '_test_pids.lst')
    train_patch_lst = []
    for pid in train_pids:
        p_patch_dirs = sorted(os.listdir(os.path.join(input_dir, pid)))
        p_patch_dirs = [d for d in p_patch_dirs if os.path.isdir(os.path.join(input_dir, pid, d))]
        p_patch_dirs = folder_selected(prefix, p_patch_dirs)
        for d in p_patch_dirs:
            patches = sorted(os.listdir(os.path.join(input_dir, pid, d)))
            if len(patches) > 5000:
                patches = random.sample(patches, k=5000)
            train_patch_lst.extend([os.path.join(pid, d, patch) for patch in patches])
    test_patch_lst = []
    for pid in test_pids:
        p_patch_dirs = sorted(os.listdir(os.path.join(input_dir, pid)))
        p_patch_dirs = [d for d in p_patch_dirs if os.path.isdir(os.path.join(input_dir, pid, d))]
        p_patch_dirs = folder_selected(prefix, p_patch_dirs)
        for d in p_patch_dirs:
            patches = sorted(os.listdir(os.path.join(input_dir, pid, d)))
            if len(patches) > 100:
                patches = random.sample(patches, k=100)
            test_patch_lst.extend([os.path.join(pid, d, patch) for patch in patches])
    lst_to_save(train_patch_lst, os.path.join(out_dir, prefix+'_train_patches.lst'))
    lst_to_save(test_patch_lst, os.path.join(out_dir, prefix+'_test_patches.lst'))

if __name__ == '__main__':

    input_dir = '/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/predata_level2_margin_tdot9_sttd'
    out_dir = input_dir
    base_name = os.path.basename(input_dir)

    prefixes = ['situ', 'lymph', 'coexist']
    for prefix in prefixes:
        gen_single_lst(prefix, input_dir, out_dir)
        print(f"{prefix} data list completed!")

    # input_dir = '/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/predata_level2_margin_tdot9_sttd'
    # out_dir = input_dir

    # pids = sorted([p for p in os.listdir(input_dir) if p.startswith('13')])
    # for p in pids:
    #     all_files = os.listdir(osp.join(input_dir, p))
    #     all_folders = [f for f in all_files if osp.isdir(osp.join(input_dir, p, f))]
    #     for f in all_folders:
    #         patches = sorted(os.listdir(osp.join(input_dir, p, f)))
    #         patches = [osp.join(p, f, patch) for patch in patches]
    #         lst_to_save(patches, osp.join(input_dir, p, f+'.lst'))

