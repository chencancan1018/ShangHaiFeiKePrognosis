import os 
import json
import numpy as np
import pandas as pd

def lst_to_save(pids_lst, outpath):
    with open(outpath, 'w') as f:
        for p in pids_lst:
            f.writelines(p+'\n')

def read_contours_from_json(json_path, level=4):
    contours = []
    downsample = pow(2, level)
    with open(json_path, 'r') as f:
        anno_dict = json.load(f)
        labels = anno_dict['GroupModel']['Labels']
        for i in range(len(labels)):
            label = labels[i]
            coords = label['Coordinates']
            cnt = []
            for coord in coords:
                if isinstance(coord, dict):
                    x = coord['X']
                    y = coord['Y']
                elif isinstance(coord, str):
                    x = float(coord.split(',')[0])
                    y = float(coord.split(',')[1])
                else:
                    raise TypeError(f"coord has wrong type: {type(coord)}")
                x /= downsample
                y /= downsample
                cnt.append([[x, y]])
            contours.append(np.array(cnt, dtype=np.int32))
    return contours

if __name__ == "__main__":
    wsi_dir = "/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/data/WSI"
    wsi_pids = sorted([p for p in os.listdir(wsi_dir) if p.startswith('13')])
    all_pids = [p.split(' ')[0] for p in wsi_pids]
    
    in_situ_pids = []
    for p in wsi_pids:
        p_name = p.split(' ')[0]
        p_path = os.path.join(wsi_dir, p)
        wsi_files = sorted(os.listdir(p_path))
        situ_wsi_files = [w for w in wsi_files if (w.startswith('A') and w.endswith('.svs'))]
        situ_json_files = [w for w in wsi_files if (w.startswith('A') and w.endswith('.json'))]
        cnts = []
        for f in situ_json_files:
            cnts.extend(read_contours_from_json(os.path.join(p_path, f)))
        if len(situ_wsi_files) > 0 and len(cnts) > 0:
            in_situ_pids.append(p_name)
    in_situ_pids = sorted(in_situ_pids)
    lst_to_save(in_situ_pids, 'all_situ_pids.lst')
    no_situ_pids = sorted(list(set(all_pids) - set(in_situ_pids)))
    print('no_situ_pids: ', len(in_situ_pids), len(no_situ_pids), no_situ_pids)
    
    lymph_pids = []
    for p in wsi_pids:
        p_name = p.split(' ')[0]
        p_path = os.path.join(wsi_dir, p)
        wsi_files = sorted(os.listdir(p_path))
        lymph_wsi_files = [w for w in wsi_files if (not w.startswith('A')) and w.endswith('.svs')]
        lymph_json_files = [w for w in wsi_files if (not w.startswith('A')) and w.endswith('.json')]
        cnts = []
        for f in lymph_json_files:
            cnts.extend(read_contours_from_json(os.path.join(p_path, f)))
        if len(lymph_wsi_files) > 0 and len(cnts) > 0:
            lymph_pids.append(p_name)
    lymph_pids = sorted(lymph_pids)
    lst_to_save(lymph_pids, 'all_lymph_pids.lst')
    no_lymph_pids = sorted(list(set(all_pids) - set(lymph_pids)))
    print('no_lymph_pids: ', len(lymph_pids), len(no_lymph_pids), no_lymph_pids)
    
    coexist_pids = sorted(list(set(in_situ_pids) & set(lymph_pids)))
    lst_to_save(coexist_pids, 'all_coexist_pids.lst')
    print('coexiste_pids: ', len(coexist_pids))
    
    
    
    
    
    
    
    
            
            
    
        
    
    
    