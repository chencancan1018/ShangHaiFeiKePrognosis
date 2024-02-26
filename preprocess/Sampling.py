import os
import cv2
import utils
import random 
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

class TiledSampling():
    """
    The whole slide image is cropped by tiled-sampling method. 
    Parameters
    ----------
    roi:  dict. The region of interest of tissue on the whole slide image.
    mask: array. The binary mask of the WSI corresponding to the roi.
    patch_sise: tuple. The size of patch need to be cropped.
    target_level: integer. The target level of WSI about the cropping level of patches.
    reverse_thresh: value less than 1. The threshold if the cropped patch is kept or abondoned.
    """
    def __init__(
        self, 
        roi, anno_mask, roi_mask,
        margin=300,
        patch_size=(256,256), 
        target_level=0, 
        reserve_thesh=0.7
    ):
        print('roi: ', roi)
        self.roi_level = roi['level']
        self.target_level = target_level
        self.bbox = [roi['start_x'], roi['start_y'], roi['w'], roi['h']]
        self.patch_size = patch_size
        self.reserve_thesh = reserve_thesh
        self.margin = int(margin / pow(2, self.roi_level))
        self.anno_mask = (anno_mask > 0) * 1
        self.roi_mask = (roi_mask > 0) * 1
        self.find_margin_area()
        
    def find_margin_area(self):
        
        # os.makedirs('./test', exist_ok=True)
        # cv2.imwrite(
        #     './test/anno_mask.jpg', 
        #     cv2.resize((self.anno_mask * 255).astype(np.uint8), dsize=None, fx=0.25, fy=0.25,interpolation=cv2.INTER_LINEAR)
        #     )
        
        h, w = self.roi_mask.shape
        if min(self.bbox[-2], self.bbox[-1]) < self.margin:
            dilate_mask = binary_dilation(self.anno_mask, iterations=self.margin)
            final_mask = dilate_mask * self.roi_mask
        else:
            dilate_mask = binary_dilation(self.anno_mask, iterations=self.margin)
            eros_mask = binary_erosion(self.anno_mask, iterations=self.margin)
            final_mask = dilate_mask * (1 - eros_mask) * self.roi_mask
        self.anno_mask = final_mask
        start_x = max(0, self.bbox[0] - self.margin)
        start_y = max(0, self.bbox[1] - self.margin)
        bbox_w = min(self.bbox[2] + 2 * self.margin, w)
        bbox_h = min(self.bbox[3] + 2 * self.margin, h)
        self.bbox = [start_x, start_y, bbox_w, bbox_h]
        
        # cv2.imwrite(
        #     './test/anno_mask_margin.jpg', 
        #     cv2.resize((self.anno_mask * 255).astype(np.uint8), dsize=None, fx=0.25, fy=0.25,interpolation=cv2.INTER_LINEAR)
        #     )

    def get_patches(self):
        patch_size = list(self.patch_size)
        downsample = pow(2, (self.roi_level - self.target_level))
        stride = [int(p / downsample) for p in patch_size]
        stride = tuple(stride)
        bbox = self.bbox
        coords_x = [bbox[0] + i * stride[0] for i in range(int(np.ceil(bbox[2] / stride[0])))]
        coords_y = [bbox[1] + i * stride[1] for i in range(int(np.ceil(bbox[3] / stride[1])))]
        patches = []
        h, w = self.roi_mask.shape
        for x in coords_x:
            for y in coords_y:
                # img.shape or mask.shape is opposite to tif/cv file. That's to say, if bbox: (w, h)-(x, y), then mask: (h, w)-(y, x).
                # Additionally, the location direction of OpenSile is consisted with bbox coordinates and direction
                if x + stride[0] > w:
                    x = max(0, w - stride[0])
                if y + stride[1] > h:
                    y = max(0, h - stride[1])
                patches.append([x, y, stride[0], stride[1]])
        return patches
    
    def filter_patches(self, patches):
        # img.shape or mask.shape is opposite to tif/cv file. That's to say, if bbox: (w, h)-(x, y), then mask: (h, w)-(y, x).
        # Additionally, the location direction of OpenSile is consisted with bbox coordinates and direction
        out = []
        for p in patches:
            p_mask = self.anno_mask[p[1]:(p[1]+p[3]), p[0]:(p[0]+p[2])]
            w, h = p_mask.shape
            p_mask = (p_mask > 0) * 1
            ratio = p_mask[p_mask > 0].sum() / (w*h)
            if ratio >= self.reserve_thesh:
                p_dict= {}
                p_dict['start_x'] = p[0]
                p_dict['start_y'] = p[1]
                p_dict['w'] =  p[2]
                p_dict['h'] =  p[3]
                p_dict['level'] = self.roi_level
                out.append(p_dict)
        return out
    
    def apply(self):
        patches_before = self.get_patches()
        patches_after = self.filter_patches(patches_before)
        return patches_before, patches_after
    
class RandomSampling():
    """
    The whole slide image is cropped by randomly sampling method. 
    Parameters
    ----------
    roi:  dict. The region of interest of tissue on the whole slide image.
    mask: array. The binary mask of the WSI corresponding to the roi.
    patch_sise: tuple. The size of patch need to be cropped.
    target_level: integer. The target level of WSI about the cropping level of patches.
    reverse_thresh: value less than 1. The threshold if the cropped patch is kept or abondoned.
    """
    def __init__(
        self, roi, anno_mask, roi_mask, num, 
        margin=300,
        patch_size=(256,256), 
        target_level=0, 
        reserve_thesh=0.7
    ):
        print('roi: ', roi)
        self.roi_level = roi['level']
        self.target_level = target_level
        self.bbox = [roi['start_x'], roi['start_y'], roi['w'], roi['h']]
        self.patch_size = patch_size
        self.num = num
        self.reserve_thesh = reserve_thesh
        self.margin = margin
        self.anno_mask = (anno_mask > 0) * 1
        self.roi_mask = (roi_mask > 0) * 1
        self.find_margin_area()
        
    def find_margin_area(self):
        w, h = self.roi_mask.shape
        if min(self.bbox[-2], self.bbox[-1]) < self.margin:
            dilate_mask = binary_dilation(self.anno_mask, iterations=self.margin)
            final_mask = dilate_mask * self.roi_mask
        else:
            dilate_mask = binary_dilation(self.anno_mask, iterations=self.margin)
            eros_mask = binary_erosion(self.anno_mask, iterations=self.margin)
            final_mask = dilate_mask * (1 - eros_mask) * self.roi_mask
        self.anno_mask = final_mask
        start_x = max(0, self.bbox[0] - self.margin)
        start_y = max(0, self.bbox[1] - self.margin)
        bbox_w = min(self.bbox[2] + 2 * self.margin, w)
        bbox_h = min(self.bbox[3] + 2 * self.margin, h)
        self.bbox = [start_x, start_y, bbox_w, bbox_h]

    def get_patches(self):
        patch_size = list(self.patch_size)
        downsample = pow(2, (self.roi_level - self.target_level))
        stride = [int(p / downsample) for p in patch_size]
        stride = tuple(stride)
        bbox = self.bbox
        patches = []
        h, w = self.roi_mask.shape
        iteration = 0 
        while (len(patches) < self.num) and (iteration < 5000):
            iteration += 1
            x = random.randint(bbox[0], bbox[0] + bbox[2])
            y = random.randint(bbox[1], bbox[1] + bbox[3])
            # img.shape or mask.shape is opposite to tif/cv file. That's to say, if bbox: (w, h)-(x, y), then mask: (h, w)-(y, x).
            # Additionally, the location direction of OpenSile is consisted with bbox coordinates and direction
            if x + stride[0] > w:
                x = max(0, w - stride[0])
            if y + stride[1] > h:
                y = max(0, h - stride[1])
            p = [x, y, stride[0], stride[1]]
            if self.filter_patch(p):
                p_dict= {}
                p_dict['start_x'] = p[0]
                p_dict['start_y'] = p[1]
                p_dict['w'] =  p[2]
                p_dict['h'] =  p[3]
                p_dict['level'] = self.roi_level
                patches.append(p_dict)
        return patches
    
    def filter_patch(self, p):
        # img.shape or mask.shape is opposite to tif/cv file. That's to say, if bbox: (w, h)-(x, y), then mask: (h, w)-(y, x).
        # Additionally, the location direction of OpenSile is consisted with bbox coordinates and direction
        p_mask = self.anno_mask[p[1]:(p[1]+p[3]), p[0]:(p[0]+p[2])]
        w, h = p_mask.shape
        p_mask = (p_mask > 0) * 1
        ratio = p_mask[p_mask > 0].sum() / (w*h)
        if ratio >= self.reserve_thesh:
            return True
        else:
            return False
    
    def apply(self):
        patches = self.get_patches()
        return patches

if __name__ == '__main__':
    tif_path = '/home/tx-deepocean/Data1/data1/workspace_ccc/PathoCTEGFR/data/internal/pathologyData/pathology12/201313778.tif'
    outdir = './test'
    os.makedirs(outdir, exist_ok=True)
    import pipeline
    from WSI import WSI

    rois = pipeline.get_tissue_rois(tif_path, outdir, level=4)
    wsi_name = os.path.basename(tif_path)[:-4]
    wsi = WSI(tif_path)
    bgr_image = wsi.get_wsi_data(level=4)
    mask, mask_close, _ = utils.ROI_detection(bgr_image, upper=(230,230,230), size1=(7,7), size2=(5,5))
    cv2.imwrite(
        os.path.join(outdir, wsi_name+'_ori_mask.jpg'), 
        cv2.resize(mask_close, dsize=None, fx=0.25, fy=0.25,interpolation=cv2.INTER_LINEAR)
    )
    for roi in rois[:1]:
        # sample = Sampling(roi, mask_close, patch_size=(1024*4,1024*4))
        sample = TiledSampling(roi, mask_close, patch_size=(256,256))
        patches_before, patches_after = sample.apply()
        print(len(patches_before), len(patches_after))
        for i, p in enumerate(patches_after):
            print(p)
        #     cv2.rectangle(bgr_image, (p['start_x'], p['start_y']), (p['start_x']+ p['w'], p['start_y']+p['h']), color=(255,0,0), thickness=(i+1))
        #     cv2.imwrite(
        #         os.path.join(outdir, wsi_name+'_image_'+str(i+1)+'.jpg'), 
        #         cv2.resize(bgr_image, dsize=None, fx=0.25, fy=0.25,interpolation=cv2.INTER_LINEAR)
        #     )
        #     downsample = pow(2, p['level'])
        #     location = (p['start_x'] * downsample, p['start_y'] * downsample)
        #     size = (p['w'], p['h'])
        #     p_image = wsi.read_region(location, size, p['level'])
        #     cv2.imwrite(
        #         os.path.join(outdir, wsi_name+'_patch_'+str(i+1)+'.jpg'), 
        #         cv2.resize(p_image, dsize=None, fx=0.25, fy=0.25,interpolation=cv2.INTER_LINEAR)
        #     )

