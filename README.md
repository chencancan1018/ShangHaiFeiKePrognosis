# 数据预处理
- train/test数据划分：preprocess/data_split/split.py
- 生成ROI附近(内3mm+外3mm)pathces: preprocess/pipeline.py
- patches 染色标准化：preprocess/stain_normalization/stain_normal_single_process.py
- 生成patches train/test数据集:preprocess/data_split/generate_patch_lst.py

# AI模型构建

- 模型训练：code_2cates/train/tools/dist_train.sh
- 模型推理：code_2cates/example/main_2D.py
