# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/slpd/'
# data_root = 'data/ufpr_alpr/'

img_size = (800,800)
affine_scale = 0.5
mixup_prob = 0.1
max_aspect_ratio = 100
backend_args = None

metainfo = {
    'classes': ('LP', ),
    'palette': [
        (255, 0, 0),
    ]
}

pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True)
]

mosaic_affine_transform = [
    dict(
        type='Mosaic',
        img_scale=img_size,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_size[0] // 2, -img_size[1] // 2),
        border_val=(114, 114, 114))
]

albu_train_transforms = [
    dict(type='Blur', p=0.1),
    dict(type='MedianBlur', p=0.1),
    dict(type='ToGray', p=0.1),
    dict(type='CLAHE', p=0.1)
]

last_transform = [
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='Resize', scale=img_size, keep_ratio=False),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_pipeline = [
    * pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_size,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        # img_scale is (width, height)
        border=(-img_size[0] // 2, -img_size[1] // 2),
        border_val=(114, 114, 114)),
    *last_transform
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_size, keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='base_CocoMetric',
    ann_file=data_root + 'annotations/instances_test2017.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args
)
test_evaluator = val_evaluator
