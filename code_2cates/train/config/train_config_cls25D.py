trainner = dict(type='Trainner_', runner_config=dict(type='EpochBasedRunner'))

# 模型训练过程中使用的部分参数
in_ch = 3
patch_size = [32, 256, 256] 


# 此实例表明network为类SegNetwork,backbone为ResUnet,head为SegHead
aux_loss = True
model = dict(
    type='ClsNetwork25D',
    backbone='inception_v3',
    in_ch=in_ch,
    num_classes=3,
    loss_weight=(1, 3),
    pretrained=True,
    patch_size=patch_size,
    apply_sync_batchnorm=True,
)
train_cfg = None
test_cfg = None

import numpy as np

data = dict(
    imgs_per_gpu=24,
    workers_per_gpu=2,
    shuffle=True,
    drop_last=False,
    dataloader=dict(
        type='SampleDataLoader',
        source_batch_size=2,
        source_thread_count=1,
        source_prefetch_count=1,
    ),
    train=dict(
        type='Cls25DSampleDataset', 
        dst_list_file='/home/tx-deepocean/Data1/data1/workspace_ccc/PathoCTEGFR/data/internal/preprocess/split/train.lst',
        data_root='/home/tx-deepocean/Data1/data1/workspace_ccc/PathoCTEGFR/data/internal/preprocess/pathology/predata_level1',
        patch_size=patch_size,
        # target_level=0,
        target_level=1,
        sample_frequent=50,
        pipelines=[],
    ),
    val=dict(
        type='Cls25DSampleDataset', 
        dst_list_file='/home/tx-deepocean/Data1/data1/workspace_ccc/PathoCTEGFR/data/internal/preprocess/split/test.lst',
        data_root='/home/tx-deepocean/Data1/data1/workspace_ccc/PathoCTEGFR/data/internal/preprocess/pathology/predata_level1',
        patch_size=patch_size,
        # target_level=0,
        target_level=1,
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
total_epochs = 100
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
work_dir = './checkpoints/results/inceptionv3_25d_level1_1030'

workflow = [('train', 1), ('val', 1)]

