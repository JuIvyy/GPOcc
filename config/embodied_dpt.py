optimizer_wrapper = dict(
    optimizer = dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1)}
    ),
)
grad_max_norm = 35
amp = False
seed = 1
print_freq = 1
eval_freq = 10
max_epochs = 10
load_from = 'work_dirs/mono_dpt_bin16/latest.pth'
find_unused_parameters = True
track_running_stats = True
flag_depthanything_as_gt = False

ignore_label = 0
empty_idx = 12   # 0 ignore, 1~11 objects, 12 empty
cls_dims = 13

pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
scale_range = [0.01, 0.08]
image_size = [480, 640]
resize_lim = [1.0, 1.0] 
num_frames = 1
offset = 0
grad_frames = None

_dim_ = 96
num_cams = 1
num_heads = 3
num_levels = 4
num_anchor = 16200
num_anchor_init = 8100
num_cross_layer = 3
num_self_layer = 3
num_decoder_fillhead = 2
semantics_activation = 'identity'
use_camera_embed = False

anchor_encoder = dict(
    type='SparseGaussian3DEncoder',
    embed_dims=_dim_,
    semantic_dim=cls_dims-1,
)

refine_layer = dict(
    type='SparseGaussian3DDeltaRefinementModule',
    embed_dims=_dim_,
    pc_range=pc_range,
    scale_range=scale_range,
    restrict_xyz=True,
    unit_xyz=[0.1, 0.1, 0.06], 
    refine_manual=[0, 1, 2],
    semantic_dim=cls_dims-1,
    semantics_activation=semantics_activation,
)

spconv_layer=dict(
    type='SparseConv3D',
    in_channels=_dim_,
    embed_channels=_dim_,
    pc_range=pc_range,
    grid_size=[0.08]*3, 
    kernel_size=3,
)

model = dict(
    type='VGGTGaussianSegmentorOnline',
    frozen_backbone=False,
    freeze_blocks=0, # total 24
    flag_depthbranch=False,
    flag_depthanything_as_gt=flag_depthanything_as_gt,
    use_depthanything=True,
    render_feat=False,
    detach_local=True,
    num_bins=16,

    semantic_dim=cls_dims-1,

    merge_kernel_dist_thresh=0.02,
    opacities_threshold=0.01,
    opa_to_thickness=False,
    use_markley_avg_quat=False,
    use_log_avg=False,
    weight_use_p=False,
    weight_use_t=True,
    temporal_alpha=0.1,

    cuda_kwargs=dict(
        scale_multiplier=3,
        H=60, W=60, D=36,
        pc_min=[-51.2, -51.2, -5.0],
        grid_size=0.08),
    global_cuda_kwargs=dict(
        scale_multiplier=3,
        H=200, W=220, D=90,
        pc_min=[-51.2, -51.2, -5.0],
        grid_size=0.08),
)


loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='FocalLoss',
            weight=100.0, 
            gamma=2.0,
            alpha=0.25,
            cls_freq=[5080655412, 722756, 44793226, 41084591, 3416464, 21897101, 10609339, 13846320, 23470172, 263393, 30949122, 9871618, 3196722886],
            ignore_label=ignore_label,
            input_dict={
                'pred': 'ce_input',
                'target': 'ce_label',
                'fov_mask': 'fov_mask'}),
        dict(
            type='LovaszLoss',
            weight=1.0,
            ignore_label=ignore_label,
            input_dict={
                'lovasz_input': 'ce_input',
                'lovasz_label': 'ce_label',
                'fov_mask': 'fov_mask'}),
        dict(
            type='Sem_Scal_Loss',
            weight=1.0,
            ignore_label=ignore_label,
            sem_cls_range=[1, 12],
            input_dict={
                'pred': 'ce_input',
                'ssc_target': 'ce_label',
                'fov_mask': 'fov_mask'}),
        dict(
            type='Geo_Scal_Loss',
            weight=1.0,
            empty_idx=empty_idx,
            ignore_label=ignore_label,
            input_dict={
                'pred': 'ce_input',
                'ssc_target': 'ce_label',
                'fov_mask': 'fov_mask'}),
    ]
)

data_path = './data/occscannet' # path/to/your/data/occscannet

train_dataset_config = dict(
    type='Scannet_Online_SceneOcc_Dataset',
    # data_path = data_path,
    num_frames = num_frames,
    # offset = offset,
    empty_idx = empty_idx,
    phase='train',
    num_pts=num_anchor_init,
    data_tag='base', # 'mini' for mini-set
    vggt_image_preprocess=True,
)

val_dataset_config = dict(
    type='Scannet_Online_SceneOcc_Dataset',
    # data_path = data_path,
    num_frames = num_frames,
    # offset = offset,
    empty_idx=empty_idx,
    phase='test',
    num_pts=num_anchor_init,
    data_tag='base', # 'mini' for mini-set
    vggt_image_preprocess=True,
)

train_wrapper_config = dict(
    type='Scannet_Online_SceneOcc_DatasetWrapper_VGGT',
    final_dim = [480, 640], 
    resize_lim = resize_lim,
    phase='train', 
)

val_wrapper_config = dict(
    type='Scannet_Online_SceneOcc_DatasetWrapper_VGGT',
    final_dim = [480, 640],
    resize_lim = resize_lim,
    phase='test', 
    # phase='train', # for vis
)

train_loader_config = dict(
    batch_size = 2,
    shuffle = True,
    num_workers = 5,
)

val_loader_config = dict(
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
)
