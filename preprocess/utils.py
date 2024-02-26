import os
import cv2
import numpy as np
from skimage import measure

# openslide: PIL.Image, RGB
# opencv: BGR
def rgba2bgr(rgba_arr):
    rgba_arr = np.array(rgba_arr)
    bgr_arr = cv2.cvtColor(rgba_arr, cv2.COLOR_RGBA2BGR)
    return bgr_arr

def rgba2rgb(rgba_arr):
    rgba_arr = np.array(rgba_arr)
    rgb_arr = cv2.cvtColor(rgba_arr, cv2.COLOR_RGBA2RGB)
    return rgb_arr

def Binary_mask_generation(lower, upper, bgr_image):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array(lower)
    upper_red = np.array(upper)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    return mask

def Morphology_Exchange(size1, size2, mask, method='close_open'):
    kernel1 = np.ones(size1, dtype=np.uint8)
    kernel2 = np.ones(size2, dtype=np.uint8)
    if method == 'close_open' :
        mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
        mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel2)
        return (mask_open, mask_close)
    else:
        mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
        mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel2)
        return (mask_open, mask_close)

def ROI_detection(
        bgr_image, 
        lower=(20,20,20), 
        upper=(230,230,230), 
        size1=(5,5), 
        size2=(5,5),
    ):
    '''HSV_threshold  method to obtain ROI
        Args:
            bgr_image: images of BGR
            lower: threshold
            upper: threshold
            size1: tuple, kernel of close_open operation
            size2: tuple, kernel of close_open operation
        returns:
            mask: HSV_images which through the HSV_threshold
            image_open: mask which through the open operation
            image_close: mask which through the close operation
    '''
    ## convert
    mask = Binary_mask_generation(lower, upper, bgr_image)
    mask_open, mask_close = Morphology_Exchange(size1, size2, np.array(mask))
    return (mask, mask_close, mask_open)

def draw_contours(bgr_image):
    _, mask, _ = ROI_detection(bgr_image)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr_image, contours, -1, (255,0,0), 3) 
    #-1 表示绘制所有轮廓，3表示轮廓线的厚度，-1(cv2.FILLED)为填充模式

    bboxes = [cv2.boundingRect(c) for c in contours]
    for i, bbox in enumerate(bboxes):
        x = int(bbox[0])
        y = int(bbox[1])
        cv2.rectangle(bgr_image, (x, y), (x + bbox[2], y + bbox[3]), color=(0, 0, 255),thickness=2)
    return bgr_image, bboxes

def make_mask(img, contours):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, -1)
    return mask

def fill_hole(bin_mask):
    assert len(bin_mask.shape) == 2
    labeled = measure.label((bin_mask > 0))
    regions = measure.regionprops(labeled)
    hole = np.zeros_like(labeled)
    for r in regions:
        hole[r.bbox[0]:r.bbox[2], r.bbox[1]:r.bbox[3]] = r.filled_image
    bin_mask[hole > 0] = 255
    return bin_mask

def contours_sorted(contours, reverse=True):
    areas = [cv2.contourArea(c) for c in contours]
    idxes = np.argsort(areas)
    if reverse:
        contours = [contours[i] for i in idxes[::-1]]
    else:
        contours = [contours[i] for i in idxes]
    return contours

def contours_filter(contours, min_thresh=0.1):
    l = len(contours)
    if l > 0:
        contours = contours_sorted(contours)
        out = [contours[0]]
        max_area = cv2.contourArea(contours[0])
        if l > 1:
            for c in contours[1:]:
                c_area = cv2.contourArea(c)
                ratio = c_area / (max_area + 1)
                if ratio > min_thresh:
                    out.append(c)
        return out
    else:
        raise Exception('The contours is empty!')
        

