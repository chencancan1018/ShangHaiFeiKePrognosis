import os
import cv2
import time
import spams
import traceback
import numpy as np
from vahadane  import VahadaneNormalizer
from macenko import MacenkoNormalizer
from multiprocessing import Pool, Process

from utils import misc_utils as mu

def get_stain_matrix(I, threshold=0.8, lamda=0.1):
    """
    Get 2x3 stain matrix. First row H and second row E.
    See the original paper for details.
    Also see spams docs.
    :param I:
    :param threshold:
    :param lamda:
    :return:
    """
    print('>>>>>>>>>>>>>>>>>>>>>>>debug00')
    mask = mu.notwhite_mask(I, thresh=threshold).reshape((-1,))
    print('>>>>>>>>>>>>>>>>>>>>>>>debug11')
    OD = mu.RGB_to_OD(I).reshape((-1, 3))
    print('>>>>>>>>>>>>>>>>>>>>>>>debug22')
    OD = OD[mask]
    print('>>>>>>>>>>>>>>>>>>>>>>>debug33')
    dictionary = spams.trainDL(OD.T, K=2, lambda1=lamda, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False).T
    print('>>>>>>>>>>>>>>>>>>>>>>>debug44')
    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
        print('>>>>>>>>>>>>>>>>>>>>>>>debug55')
    dictionary = mu.normalize_rows(dictionary)
    print('>>>>>>>>>>>>>>>>>>>>>>>debug66')
    return dictionary

def get_concentrations(I, stain_matrix, lamda=0.01):
    """
    Get the concentration matrix. Suppose the input image is H x W x 3 (uint8). Define Npix = H * W.
    Then the concentration matrix is Npix x 2 (or we could reshape to H x W x 2).
    The first element of each row is the Hematoxylin concentration.
    The second element of each row is the Eosin concentration.

    We do this by 'solving' OD = C*S (Matrix product) where OD is optical density (Npix x 3),\
    C is concentration (Npix x 2) and S is stain matrix (2 x 3).
    See docs for spams.lasso.

    We restrict the concentrations to be positive and penalise very large concentration values,\
    so that background pixels (which can not easily be expressed in the Hematoxylin-Eosin basis) have \
    low concentration and thus appear white.

    :param I: Image. A np array HxWx3 of type uint8.
    :param stain_matrix: a 2x3 stain matrix. First row is Hematoxylin stain vector, second row is Eosin stain vector.
    :return:
    """
    print('<<<<<<<<<<<<<<<<<<<<<<debug00')
    OD = mu.RGB_to_OD(I).reshape((-1, 3))  # convert to optical density and flatten to (H*W)x3.
    print('<<<<<<<<<<<<<<<<<<<<<<debug11')
    return spams.lasso(OD.T, D=stain_matrix.T, mode=2, lambda1=lamda, pos=True).toarray().T

def transform(I, stain_matrix_target):
    """
    Transform an image
    :param I:
    :return:
    """
    print('==============debug00')
    I = mu.standardize_brightness(I)
    print('==============debug11')
    stain_matrix_source = get_stain_matrix(I)
    print('==============debug22')
    source_concentrations = get_concentrations(I, stain_matrix_source)
    print('==============debug33')
    return (255 * np.exp(-1 * np.dot(source_concentrations, stain_matrix_target).reshape(I.shape))).astype(
        np.uint8)

def trans_func(jpg_path, jpg, cur_out_dir, stain_matrix_target):
    try:
        print('debug00')
        out_path = os.path.join(cur_out_dir, jpg)
        print('debug11')
        bgr_img = cv2.imread(jpg_path)
        print('debug22')
        bgr_img2 = transform(bgr_img, stain_matrix_target)
        print('debug33')
        bgr_img2 = bgr_img
        print('debug44')
        cv2.imwrite(out_path, bgr_img2)
        print('debug55')
    except:
        traceback.print_exc()

def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # p = np.percentile(img, 90)
    # img = np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)
    return img

if __name__ == "__main__":
    sttd_path = '/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/predata_level2_margin_tdot9/refer_bgr_img.jpg'
    sttd_img = read_image(sttd_path)
    stain_method = VahadaneNormalizer()
    stain_method.fit(sttd_img)
    stain_matrix_target = stain_method.return_stain_matrix()

    patch_dir = "/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/predata_level2_margin_tdot9"
    out_dir = patch_dir
    pids = os.listdir(patch_dir)
    pids = sorted([str(p) for p in pids if p.startswith('13')])[2:3]
    for p in pids:
        pid_info = []
        all_files = sorted(os.listdir(os.path.join(patch_dir, p)))
        patch_folders = [f for f in all_files if os.path.isdir(os.path.join(patch_dir, p, f))]
        for patch_folder in patch_folders:
            start_time = time.time()
            cur_out_dir = os.path.join(out_dir, p, patch_folder+'_sttd')
            os.makedirs(cur_out_dir, exist_ok=True)
            jpgs = sorted(os.listdir(os.path.join(patch_dir, p, patch_folder)))[:60]
            pool = Pool(processes=10)
            for jpg in jpgs:
                jpg_path = os.path.join(patch_dir, p, patch_folder, jpg)
                bgr_img_ori = cv2.imread(jpg_path)
                info = (jpg_path, jpg, cur_out_dir, stain_matrix_target)
                pool.apply_async(trans_func, info)   
            pool.close()
            pool.join()
            requests = [req.get() for req in requests]
            end_time = time.time()
            print(f"{p}: {patch_folder} completed and cost time: {end_time-start_time} s!")

                

    

