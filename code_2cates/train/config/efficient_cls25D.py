trainner = dict(type='Trainner_', runner_config=dict(type='EpochBasedRunner'))

# 模型训练过程中使用的部分参数
win_level = [-600]
win_width = [1800]
in_ch = 3
patch_size = [64, 224, 224] # for coarse- and fine-seg


# 此实例表明network为类SegNetwork,backbone为ResUnet,head为SegHead
aux_loss = True
model = dict(
    type='ClsNetwork25D',
    # backbone='convnext_nano',
    backbone='tf_efficientnet_es',
    in_ch=in_ch+1,
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
    imgs_per_gpu=8,
    workers_per_gpu=4,
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
        dst_list_file='./checkpoints/predata/train_0.lst',
        data_root='./checkpoints/predata',
        patch_size=patch_size,
        sample_frequent=10,
        win_level=win_level,
        win_width=win_width,
        pipelines=[
            dict(
                type="MonaiElasticDictTransform",
                aug_parameters={
                    "prob": 0.5,
                    "patch_size": patch_size,
                    "roi_scale": 1.0,
                    "max_roi_scale": 1.0,
                    "rot_range_x": (-np.pi/9, np.pi/9),
                    "rot_range_y": (-np.pi/9, np.pi/9),
                    "rot_range_z": (-np.pi/9, np.pi/9),
                    "rot_90": False,
                    "flip": True,
                    "bright_bias": (-0.2, 0.2),
                    "bright_weight": (-0.2, 0.2),
                    "translate_x": (-5.0, 5.0),
                    "translate_y": (-5.0, 5.0),
                    "translate_z": (-5.0, 5.0),
                    "scale_x": (-0.2, 0.2),
                    "scale_y": (-0.2, 0.2),
                    "scale_z": (-0.2, 0.2),
                    "elastic_sigma_range": (3, 5),  # x,y,z
                    "elastic_magnitude_range": (100, 200),
                },
            )
        ],
    ),
    val=dict(
        type='Cls25DSampleDataset', 
        dst_list_file='./checkpoints/predata/val_0.lst',
        data_root='./checkpoints/predata',
        patch_size=patch_size,
        sample_frequent=1,
        win_level=win_level,
        win_width=win_width,
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
work_dir = './checkpoints/results/Cls25D_fold0_efficient_0913'

workflow = [('train', 1), ('val', 1)]

