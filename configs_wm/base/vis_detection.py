# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/ubuntu/wm_datasets/vis/'
# data_root = '/home/ubuntu/wm_datasets/slpd/'
# data_root = '/home/ubuntu/wm_datasets/UFPR_ALPR/'

img_size = (1024,1024)
batch_size = 8
backend_args = None

metainfo = {
    'classes': (
        "ignored regions", "pedestrian", "people",
        "bicycle", "car", "van",
        "truck", "tricycle", "awning-tricycle",
        "bus", "motor", "others"
    ),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 142),
        (0, 0, 230), (106, 0, 228), (0, 60, 100),
        (0, 80, 100), (0, 0, 70), (0, 0, 192),
        (250, 170, 30), (100, 170, 30), (220, 220, 0),
    ]
}

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.0),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_size, keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        # ann_file='annotations/instances_test2017.json',
        # ann_file='annotations/train_sp.json',
        # ann_file='annotations/jx.json',
        # ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        # ann_file='annotations/instances_test2017.json',
        # ann_file='annotations/jx.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    # type='wm_CocoMetric',
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    # ann_file=data_root + 'annotations/instances_test2017.json',
    # ann_file=data_root + 'annotations/jx.json',
    # metric=['precision','recall', 'map50', 'map','f1'],
    metric='bbox',
    format_only=False,
    backend_args=backend_args
)
test_evaluator = val_evaluator
