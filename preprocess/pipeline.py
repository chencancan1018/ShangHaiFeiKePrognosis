import os
import cv2
import time
import json
from tqdm import tqdm
import numpy as np
import utils 
from WSI import WSI
import pandas as pd
from Sampling import TiledSampling, RandomSampling

class JsonEncoder(json.JSONEncoder):
    """Convert numpy classes to JSON serializable objects."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)

def get_tissue_rois(bgr_image, wsi_name, anno_cnts, visual_path, level=4):
    _, mask_close, _ = utils.ROI_detection(bgr_image, upper=(240,240,240), size1=(50,50))
    mask_close = utils.fill_hole(mask_close)
    cv2.imwrite(
        os.path.join(visual_path, wsi_name+'_image.jpg'), 
        cv2.resize(bgr_image, dsize=None, fx=0.25, fy=0.25,interpolation=cv2.INTER_LINEAR)
    )

    contours, _ = cv2.findContours(mask_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    if len(contours) > 0:
        contours = utils.contours_filter(contours, min_thresh=0.1)
        mask = utils.make_mask(bgr_image, contours)
        cv2.imwrite(
        os.path.join(visual_path, wsi_name+'_mask.jpg'), 
        cv2.resize(mask, dsize=None, fx=0.25, fy=0.25,interpolation=cv2.INTER_LINEAR)
        )
        
        if len(anno_cnts) > 0:
            cv2.drawContours(bgr_image, anno_cnts, -1, (255,0,0), thickness=3)
            
        for i, contour in enumerate(contours):
            bbox = cv2.boundingRect(contour)
            x = int(bbox[0])
            y = int(bbox[1])
            cv2.rectangle(bgr_image, (x, y), (x + bbox[2], y + bbox[3]), color=(0,0,255),thickness=3)
            cv2.drawContours(bgr_image, [contour], -1, (0,0,255), thickness=3) 
            roi = dict()
            roi['start_x'] = bbox[0]
            roi['start_y'] = bbox[1]
            roi['w'] = bbox[2]
            roi['h'] = bbox[3]
            roi['level'] = level
            roi['path'] = wsi_path
            rois.append(roi) 
        cv2.imwrite(
            os.path.join(visual_path, wsi_name+'_bboxes.jpg'), 
            cv2.resize(bgr_image, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        )
    else:
        print(f"No exists ROI tissue In path: {wsi_path}!")
    return rois

def get_patches_from_roi(
    roi, anno_mask, roi_mask, wsi_path, 
    margin=300, 
    patch_size=(256,256), 
    target_level=0, 
    min_pathches_num=600, 
    reserve_thesh=0.7,
):
    tiled_sample = TiledSampling(
        roi, anno_mask, roi_mask,  
        margin=margin,
        patch_size=patch_size, 
        target_level=target_level, 
        reserve_thesh=reserve_thesh
    )
    _, tiled_patches = tiled_sample.apply()
    all_patches = tiled_patches
    if len(tiled_patches) < min_pathches_num:
        rand_num = min_pathches_num - len(tiled_patches)
        rand_sample = RandomSampling(
            roi, anno_mask, roi_mask, rand_num, 
            margin=margin,
            patch_size=(256,256), 
            target_level=target_level, 
            reserve_thesh=reserve_thesh
        )
        rand_patches = rand_sample.apply()
        all_patches = all_patches + rand_patches
    for p in all_patches:
        p['wsi_path'] = wsi_path
    return all_patches

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
    wsi_dir = '/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/data/WSI'
        
    # out_path = './predata'
    out_path = './predata_level0_margin_tdot9'
    os.makedirs(out_path, exist_ok=True)
    visual_path = './visual'
    os.makedirs(visual_path, exist_ok=True)
    roi_level = 4
    target_level = 0
    
    labels_df = pd.read_csv('/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/data/labels.csv', encoding='utf8')
    labels_dict = dict()
    for i in range(labels_df.shape[0]):
        labels_dict[str(labels_df.iloc[i, 0])] = labels_df.iloc[i,-1]

    pids = sorted([p for p in os.listdir(wsi_dir) if p.startswith('13')])
    bad_cases = []
    bad_rois = []
    for pid in tqdm(pids):
        print(f"===================start to process: {pid}=====================")
        start_time = time.time()
        pid_name = pid.split(' ')[0]
        wsi_names = os.listdir(os.path.join(wsi_dir, pid))
        wsi_names = sorted(list(set([wn.split('.')[0] for wn in wsi_names])))
        wsi_paths = [os.path.join(wsi_dir, pid, wn+'.svs') for wn in wsi_names]
        cur_out_path = os.path.join(out_path, pid.split(' ')[0])
        cur_visual_path = os.path.join(visual_path, pid.split(' ')[0])
        os.makedirs(cur_out_path, exist_ok=True)
        os.makedirs(cur_visual_path, exist_ok=True)

        # 每例PID患者可能有一或多张whole-slide image
        for i, wsi_path in enumerate(wsi_paths):
            if not os.path.exists(wsi_path):
                continue
            wsi_info = dict()
            wsi_patches = []
            wsi_name = os.path.basename(wsi_path)[:-4]
            print(f"==================={pid}: {wsi_name}=====================")
            anno_path = wsi_path[:-4]+'.json'
            if os.path.exists(anno_path):
                anno_cnts = read_contours_from_json(anno_path, level=4)
            else:
                anno_cnts = []
            wsi = WSI(wsi_path)
            bgr_image = wsi.get_wsi_data(level=roi_level)
            wsi_rois = get_tissue_rois(bgr_image, wsi_name, anno_cnts, cur_visual_path, level=roi_level)
            _, mask_close, _ = utils.ROI_detection(bgr_image, upper=(230,230,230), size1=(7,7), size2=(5,5))
            anno_mask = utils.make_mask(bgr_image, anno_cnts)
            valid_mask = (anno_mask > 0) * (mask_close > 0)
            cv2.imwrite(
                os.path.join(cur_visual_path, wsi_name+'_anno_mask.jpg'), 
                cv2.resize((valid_mask * 255).astype(np.uint8), dsize=None, fx=0.25, fy=0.25,interpolation=cv2.INTER_LINEAR)
            )
            mpp = wsi.get_resolution()
            mpp = [float(v) for v in mpp]
            if mpp[0] == 0:
                mpp = [0.092045000000000002, 0.092045000000000002]
            peritumor= 3 # mm
            margin = int(peritumor * 1000 / mpp[0])
            if len(wsi_rois) > 0:
                for roi in wsi_rois:
                    r_patches = get_patches_from_roi(
                        roi, anno_mask, mask_close, wsi_path, 
                        margin=margin, 
                        target_level=target_level,
                        reserve_thesh=0.9,
                    )
                    if len(r_patches) > 0:
                        wsi_patches.extend(r_patches)
                        print(pid, wsi_name, roi, len(r_patches))
                    else:
                        bad_rois.append(roi)
            patch_savedir = os.path.join(cur_out_path, wsi_name)
            os.makedirs(patch_savedir, exist_ok=True)
            for patch in wsi_patches:
                downsample = pow(2, int(patch['level']) - target_level)
                downsample2level0 = pow(2, int(patch['level']))
                location = (int(patch['start_x'] * downsample2level0), int(patch['start_y'] * downsample2level0))
                size = (int(patch['w'] * downsample), int(patch['h'] * downsample))
                # 读取bgr三通道patch
                img = wsi.read_region(location, size, target_level) 
                cv2.imwrite(
                            os.path.join(
                                patch_savedir, 
                                wsi_name+'_'+str(location[0])+'_'+str(location[1])+'_'+str(target_level)+'_label'+str(labels_dict[pid_name])+'.jpg'
                            ), 
                            img,
                        )
            wsi_info['ROIs'] = wsi_rois
            wsi_info['level'] = 4
            wsi_info['anno_cnts'] = anno_cnts
            wsi_info['patches'] = wsi_patches
            wsi_info['label'] = labels_dict[pid_name]
            if len(wsi_rois) > 0:
                with open(os.path.join(cur_out_path, wsi_name+'.json'), 'w') as f:
                    json.dump(wsi_info, f, ensure_ascii=False, cls=JsonEncoder)
            else:
                print(f"pid {pid} has no roi tissues!")
                bad_cases.append(pid)
            # except:
            #     pass
        end_time = time.time()
        print(f"The pre-process of pid {pid} takes {round(end_time - start_time, 2)} s!")
    print('bad cases: ', bad_cases)
    print('bad rois: ', bad_rois)
        
    

    

        






