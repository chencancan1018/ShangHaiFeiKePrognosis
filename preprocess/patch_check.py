import os
import numpy as np
import pandas as pd
import json

def flat(ll):
    out = []
    for l in ll:
        if isinstance(l, list):
            out.extend(l)
        else:
            out.append(l)
    return out

if __name__ == "__main__":
    patch_dir = "/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/predata_level2_margin_tdot9_sttd"
    out_dir = patch_dir
    pids = os.listdir(patch_dir)
    pids = sorted([str(p) for p in pids if p.startswith('13')])

    labels_df = pd.read_csv('/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/data/labels.csv', encoding='utf8')
    labels_dict = dict()
    for i in range(labels_df.shape[0]):
        labels_dict[str(labels_df.iloc[i, 0])] = labels_df.iloc[i,-1]
    
    patches_info = []
    for p in pids:
        pid_info = []
        all_files = sorted(os.listdir(os.path.join(patch_dir, p)))

        # add label
        # patch_folders = [f for f in all_files if os.path.isdir(os.path.join(patch_dir, p, f))]
        # label = labels_dict[p]
        # for patch_folder in patch_folders:
        #     jpgs = sorted(os.listdir(os.path.join(patch_dir, p, patch_folder)))
        #     for jpg in jpgs:
        #         ori_path = os.path.join(patch_dir, p, patch_folder, jpg)
        #         tar_path = os.path.join(patch_dir, p, patch_folder, jpg[:-4]+'_label'+str(labels_dict[p])+'.jpg')
        #         os.rename(ori_path, tar_path)
        
        # count patches
        # json_files = [f for f in all_files if f.endswith('.json')]
        # print(p, len(json_files))
        # patch_num = 0
        # for jf in json_files:
        #     with open(os.path.join(patch_dir, p, jf), 'r') as f:
        #         wsi_info = json.load(f)
        #     jf_patches = flat(wsi_info['patches'])
        #     pid_info.append([jf[:-5], len(jf_patches)])
        #     patch_num += len(jf_patches)
        # patches_info.append([p, patch_num, pid_info])
        sttd_files = [f for f in all_files if f.endswith('_sttd')]
        patch_num = 0
        for f in sttd_files:
            patches = sorted(os.listdir(os.path.join(patch_dir, p, f)))
            patches = [p for p in patches if p.endswith('.jpg')]
            pid_info.append([f, len(patches)])
            patch_num += len(patches)
        patches_info.append([p, patch_num, pid_info])
        
    pd.DataFrame(
        patches_info, columns=['pid', 'pid patch num', 'wsi patch num']
    ).to_csv(os.path.join(out_dir, 'patches_info.csv'), index=None, encoding='utf8')
            
        