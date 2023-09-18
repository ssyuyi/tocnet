# training schedule for 1x
max_epochs =30
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

lr = 1e-4
# learning rate
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=1,
        eta_min=1e-2 * lr,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True
    ),
]
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999),
    )
)

auto_scale_lr = dict(enable=False, base_batch_size=16)
