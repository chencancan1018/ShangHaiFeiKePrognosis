import os
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

if __name__ == "__main__":
    
    labels_path = "/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/data/labels.csv"
    labels_df = pd.read_csv(labels_path, encoding='utf8')
    pids_0 = [str(labels_df.iloc[i, 0]) for i in range(labels_df.shape[0]) if labels_df.iloc[i, -1]==0]
    pids_1 = [str(labels_df.iloc[i, 0]) for i in range(labels_df.shape[0]) if labels_df.iloc[i, -1]==1]
    
    situ_pids = get_pids("all_situ_pids.lst")
    situ_pids_0 = [p for p in situ_pids if p in pids_0]
    situ_pids_1 = [p for p in situ_pids if p in pids_1]
    print(len(situ_pids), len(situ_pids_0), len(situ_pids_1))
    situ_train_pids_0 = sorted(random.sample(situ_pids_0, k=int(0.7 * len(situ_pids_0))))
    situ_train_pids_1 = sorted(random.sample(situ_pids_1, k=int(0.7 * len(situ_pids_1))))
    situ_test_pids_0 = sorted(list(set(situ_pids_0) - set(situ_train_pids_0)))
    situ_test_pids_1 = sorted(list(set(situ_pids_1) - set(situ_train_pids_1)))
    situ_train_pids = sorted(list(situ_train_pids_0 + situ_train_pids_1))
    situ_test_pids = sorted(list(situ_test_pids_0 + situ_test_pids_1))
    lst_to_save(situ_train_pids, 'situ_train_pids.lst')
    lst_to_save(situ_test_pids, 'situ_test_pids.lst')
    
    lymph_pids = get_pids("all_lymph_pids.lst")
    lymph_pids_0 = [p for p in lymph_pids if p in pids_0]
    lymph_pids_1 = [p for p in lymph_pids if p in pids_1]
    print(len(lymph_pids), len(lymph_pids_0), len(lymph_pids_1))
    lymph_train_pids_0 = sorted(random.sample(lymph_pids_0, k=int(0.7 * len(lymph_pids_0))))
    lymph_train_pids_1 = sorted(random.sample(lymph_pids_1, k=int(0.7 * len(lymph_pids_1))))
    lymph_test_pids_0 = sorted(list(set(lymph_pids_0) - set(lymph_train_pids_0)))
    lymph_test_pids_1 = sorted(list(set(lymph_pids_1) - set(lymph_train_pids_1)))
    lymph_train_pids = sorted(list(lymph_train_pids_0 + lymph_train_pids_1))
    lymph_test_pids = sorted(list(lymph_test_pids_0 + lymph_test_pids_1))
    lst_to_save(lymph_train_pids, 'lymph_train_pids.lst')
    lst_to_save(lymph_test_pids, 'lymph_test_pids.lst')

    coexist_pids = get_pids("all_coexist_pids.lst")
    coexist_pids_0 = [p for p in coexist_pids if p in pids_0]
    coexist_pids_1 = [p for p in coexist_pids if p in pids_1]
    print(len(coexist_pids), len(coexist_pids_0), len(coexist_pids_1))
    coexist_train_pids_0 = sorted(random.sample(coexist_pids_0, k=int(0.7 * len(coexist_pids_0))))
    coexist_train_pids_1 = sorted(random.sample(coexist_pids_1, k=int(0.7 * len(coexist_pids_1))))
    coexist_test_pids_0 = sorted(list(set(coexist_pids_0) - set(coexist_train_pids_0)))
    coexist_test_pids_1 = sorted(list(set(coexist_pids_1) - set(coexist_train_pids_1)))
    coexist_train_pids = sorted(list(coexist_train_pids_0 + coexist_train_pids_1))
    coexist_test_pids = sorted(list(coexist_test_pids_0 + coexist_test_pids_1))
    lst_to_save(coexist_train_pids, 'coexist_train_pids.lst')
    lst_to_save(coexist_test_pids, 'coexist_test_pids.lst')
    
    
    