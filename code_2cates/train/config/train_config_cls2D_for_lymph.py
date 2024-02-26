trainner = dict(type='Trainner_', runner_config=dict(type='EpochBasedRunner'))

# 模型训练过程中使用的部分参数
in_ch = 3
patch_size = [256, 256] 


# 此实例表明network为类SegNetwork,backbone为ResUnet,head为SegHead
aux_loss = True
model = dict(
    type='ClsNetwork2DSigHead',
    backbone='inception_v3',
    in_ch=in_ch,
    num_classes=1,
    pretrained=True,
    patch_size=patch_size,
    apply_sync_batchnorm=True,
)
train_cfg = None
test_cfg = None

import numpy as np

data = dict(
    imgs_per_gpu=500,
    workers_per_gpu=0,
    shuffle=True,
    drop_last=False,
    dataloader=dict(
        type='SampleDataLoader', 
        source_batch_size=2,
        source_thread_count=1,
        source_prefetch_count=1,
    ),
    train=dict(
        type='Cls2DSampleDataset', 
        # dst_list_file='/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/data_split/lymph_train_pids.lst',
        dst_list_file='/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/predata_level2_margin_tdot9_sttd/lymph_train_patches.lst',
        data_root='/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/predata_level2_margin_tdot9_sttd',
        patch_size=patch_size,
        # sample_frequent=5000,
        sample_frequent = 1,
        pipelines=[],
    ),
    val=dict(
        type='Cls2DSampleDataset', 
        # dst_list_file='/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/data_split/lymph_test_pids.lst',
        dst_list_file='/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/predata_level2_margin_tdot9_sttd/lymph_test_patches.lst',
        data_root='/home/tx-deepocean/workspace_ccc/Project/ShangHaiFeiKe_prognosis/preprocess/predata_level2_margin_tdot9_sttd',
        patch_size=patch_size,
        # sample_frequent=500,
        sample_frequent=1,
        pipelines=[],
    ),
)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-5)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# lr_config = dict(policy='step', warmup='linear', warmup_iters=50, warmup_ratio=1.0 / 3, step=[50, 100], gamma=0.2)
lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=50, warmup_ratio=1.0 / 3, min_lr=23e-6)

checkpoint_config = dict(interval=1)

log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

# torch.backends.cudnn.benchmark
cudnn_benchmark = False

# 推荐使用分布式训练, gpus=4表示使用gpu的数量
gpus = 4
find_unused_parameters = True
total_epochs = 10
autoscale_lr = None # 是否使用根据batch_size自动调整学习率
launcher = 'pytorch'  # ['none', 'pytorch', 'slurm', 'mpi']
dist_params = dict(backend='nccl')
log_level = 'INFO'
seed = None
deterministic = False
resume_from = None
evaluate = False
fp16 = dict(loss_scale=512.)

load_from = None
work_dir = './checkpoints/results_for_lymph_0222/'

workflow = [('train', 1), ('val', 1)]

