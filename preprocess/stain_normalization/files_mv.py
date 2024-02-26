import os
import shutil
import numpy as np


if __name__ == "__main__":
    src_dir = '/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/predata_level2_margin_tdot9'
    tgt_dir = '/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/predata_level2_margin_tdot9_sttd'

    pids = sorted(os.listdir(src_dir))
    pids = [p for p in pids if p.startswith('13')]
    # print(pids)

    for p in pids:
        cur_src_dir = os.path.join(src_dir, p)
        cur_tgt_dir = os.path.join(tgt_dir, p)
        os.makedirs(cur_tgt_dir, exist_ok=True)
        pfolders = [f for f in os.listdir(cur_src_dir) if f.endswith('sttd')]
        # print(pfolders)
        if len(pfolders) == 0:
            continue
        for pf in pfolders:
            print(p, pf, ' move')
            shutil.move(os.path.join(cur_src_dir, pf), os.path.join(cur_tgt_dir, pf))
        


    