_base_ = [
    '../_base_/models/mask_rcnn_fpn_lit_ti.py',
    '../_base_/datasets/coco_instance.py',
    # '../configs/_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    pretrained='pretrained/lit_ti.pth',
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12
