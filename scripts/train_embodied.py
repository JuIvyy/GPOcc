import os, time, argparse, os.path as osp, numpy as np
import cv2
import copy
import pickle

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import gc
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from gpocc.utils.iou_eval import IOUEvalBatch
from gpocc.utils.iou_as_iso import SSCMetrics, SSCMetricsTorch
from gpocc.utils.loss_record import LossRecord
from gpocc.utils.load_save_util import revise_ckpt, revise_ckpt_2, revise_ckpt_notddp

from gpocc.loss.depth_loss import DepthLoss                
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim.optimizer.builder import build_optim_wrapper
from mmengine.logging.logger import MMLogger
from mmengine.utils import symlink
from timm.scheduler import CosineLRScheduler
from mmengine.registry import MODELS
import open3d as o3d
import warnings
warnings.filterwarnings("ignore")
import sys
from PIL import Image

from gpocc.model.segmentor.gaussian_segmentor.gaussian_segmentor import VGGTGaussianSegmentorOnline


def pass_print(*args, **kwargs):
    pass


def is_main_process():
    if not dist.is_available():
        return True
    elif not dist.is_initialized():
        return True
    else:
        return dist.get_rank() == 0


def main(args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)

    # TODO
    if args.model_config is not None:
        assert args.evaluate
        model_cfg = Config.fromfile(args.model_config)
        cfg.model['num_bins'] = model_cfg.model['num_bins']
        # cfg.model['opacities_threshold'] = model_cfg.model['opacities_threshold']
        cfg.load_from = os.path.join(args.work_dir, 'latest.pth')

    set_random_seed(cfg.seed)
    cfg.work_dir = args.work_dir
    max_num_epochs = cfg.max_epochs
    eval_freq = cfg.eval_freq
    print_freq = cfg.print_freq
    vis_freq = cfg.get('vis_freq', 10)
    if args.vis_freq:
        vis_freq = args.vis_freq

    # init DDP
    distributed = True
    world_size = int(os.environ["WORLD_SIZE"])  # number of nodes
    rank = int(os.environ["RANK"])  # node id
    gpu = int(os.environ['LOCAL_RANK'])
    
    dist.init_process_group(
        backend="nccl", init_method=f"env://", 
        world_size=world_size, rank=rank
    )

    # dist.barrier()
    torch.cuda.set_device(gpu)

    if not is_main_process():
        import builtins
        builtins.print = pass_print

    # configure logger
    if is_main_process():
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_name = os.path.basename(args.work_dir)
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger(
        name='embodied',
        logger_name=exp_name,
        log_file=log_file,
        log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    from gpocc.model import build_model
    my_model = build_model(cfg.model)

    if cfg.flag_depthanything_as_gt:
        my_model.depthanything.requires_grad_(False)
    if hasattr(my_model, 'globalhead'):
        my_model.globalhead.requires_grad_(False)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    # logger.info(f'Model:\n{my_model}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        if cfg.get('track_running_stats', False):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[gpu],
            find_unused_parameters=find_unused_parameters)
    else:
        my_model = my_model.cuda()
    logger.info('done ddp model')
    # build dataloader
    from gpocc.dataset import build_dataloader
    train_dataset_loader, val_dataset_loader = \
        build_dataloader(
            cfg.train_dataset_config,
            cfg.val_dataset_config,
            cfg.train_wrapper_config,
            cfg.val_wrapper_config,
            cfg.train_loader_config,
            cfg.val_loader_config,
            dist=distributed,
        )

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    logger.info(f'resume from: {cfg.resume_from}')
    logger.info(f'work dir: {args.work_dir}')

    amp = cfg.get('amp', True)

    if not args.evaluate:
        # try post init weight for online model
        # my_model.module.post_init_weight()

        # get optimizer, loss, scheduler
        logger.info(f"Build optimizer and metrics")
        optimizer = build_optim_wrapper(my_model, cfg.optimizer_wrapper)
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        from loss import GPD_LOSS
        loss_func = GPD_LOSS.build(cfg.loss).cuda()
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=len(train_dataset_loader)*max_num_epochs,
            lr_min=1e-6,
            warmup_t=500, # FIXME
            warmup_lr_init=1e-6,
            t_in_epochs=False
        )

    # resume and load
    epoch = 0
    best_val_iou = 0
    best_val_miou = 0
    global_iter = 0

    if not args.evaluate and cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        logger.info(my_model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        if 'best_val_iou' in ckpt:
            best_val_iou = ckpt['best_val_iou']
        if 'best_val_miou' in ckpt:
            best_val_miou = ckpt['best_val_miou']
        global_iter = ckpt['global_iter']
        logger.info(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        logger.info(f"Load from {cfg.load_from}, ret:")
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        if not distributed:
            state_dict = revise_ckpt_notddp(state_dict)
        else:
            state_dict = revise_ckpt(state_dict)
        try:
            logger.info(my_model.load_state_dict(state_dict, strict=False))
        except:
            state_dict = revise_ckpt_2(state_dict)
            logger.info(my_model.load_state_dict(state_dict, strict=False))

    CalMeanIou = SSCMetricsTorch(n_classes=12)
    CalMeanIou_Fov = SSCMetricsTorch(n_classes=12)
    CalMeanIou_Global = SSCMetricsTorch(n_classes=12)

    scenemeta_keys = ['global_scene_dim', 'global_scene_size', 'global_labels', 'global_pts', 'global_scene_origin', 'global_mask']
    metas_tensor_keys_inv = ['name', 'cam2img', 'world2img', 'rgb_path', 'depth_path','num_depth', 'occ_mask_valid', 'img_shape', 'img_aug_matrix', 'img_depthbranch']

    only_global = False

    if args.evaluate:
        my_model.eval()
        CalMeanIou.reset()
        CalMeanIou_Fov.reset()
        CalMeanIou_Global.reset()
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        if args.vis or args.save:
            save_dir_occ = os.path.join(args.work_dir, 'vis_occ_occupancy')
            os.makedirs(save_dir_occ, exist_ok=True)
            save_dir_gauss = os.path.join(args.work_dir, 'vis_occ_gaussian')
            os.makedirs(save_dir_gauss, exist_ok=True)
            save_dir_img = os.path.join(args.work_dir, 'original_images')
            os.makedirs(save_dir_img, exist_ok=True)

        if args.save:
            save_dir = os.path.join(args.work_dir, 'embodied_pred_save')

        # num_gaussians = []
        with torch.no_grad():
            for i_iter_val, data in enumerate(val_dataset_loader):

                CalMeanIou.reset()

                for i in range(len(data)):
                    if isinstance(data[i], torch.Tensor):
                        data[i] = data[i].cuda()
                (imgs, metas, labels) = data

                vis_data = []
                if args.vis or args.save:
                    save_dir_occ_thisscene = os.path.join(save_dir_occ, metas[0]['scene_name'])
                    os.makedirs(save_dir_occ_thisscene, exist_ok=True)
                    save_dir_gauss_thisscene = os.path.join(save_dir_gauss, metas[0]['scene_name'])
                    os.makedirs(save_dir_gauss_thisscene, exist_ok=True)

                for k, v in metas[0].items():
                    if k in scenemeta_keys:
                        for meta in metas:
                            meta[k] = torch.tensor(meta[k]).cuda()

                K_Frames = len(metas[0]['monometa_list'])
                scenemetas = [meta['monometa_list'] for meta in metas]

                my_model.module.scene_init(metas)
                if isinstance(my_model.module, VGGTGaussianSegmentorOnline):
                    # reset
                    my_model.module.global_gaussians = None

                for i in range(K_Frames):
                    img = imgs[:, :, i, :, :, :].unsqueeze(2)
                    label = labels[:, i, :, :, :].unsqueeze(1)

                    my_model.module.update_global_mask(scenemetas, frame_idx=i)

                    with torch.cuda.amp.autocast(enabled=amp):
                        result_dict, global_result_dict = my_model(
                            scenemeta=scenemetas,
                            imgs=img,
                            metas=[m[i] for m in scenemetas],
                            points=None,
                            label=label,
                            frame_idx=i,
                            grad_frames=cfg.grad_frames,
                            test_mode=False,
                            only_global=only_global)

                    assert len(global_result_dict['ce_label']) == 1
                    voxel_predict = global_result_dict['ce_input'][0].argmax(dim=0).long() # [1, 60, 60, 36]
                    voxel_label = global_result_dict['ce_label'][0].long() # [1, 60, 60, 36]

                    voxel_predict[voxel_predict == 0] = 255
                    voxel_predict[voxel_predict == 12] = 0
                    voxel_label[voxel_label == 0] = 255
                    voxel_label[voxel_label == 12] = 0
                    CalMeanIou.add_batch(voxel_predict, voxel_label)

                    if not only_global:

                        fov_voxel_predict = result_dict['ce_input'].argmax(dim=1).long() # [1, 60, 60, 36]
                        fov_voxel_label = result_dict['ce_label'].long() # [1, 60, 60, 36]

                        this_fov_mask = metas[0]['monometa_list'][i]['fov_mask'].unsqueeze(0).bool()
                        fov_voxel_predict = fov_voxel_predict[this_fov_mask].unsqueeze(0)
                        fov_voxel_label = fov_voxel_label[this_fov_mask].unsqueeze(0)

                        fov_voxel_predict[fov_voxel_predict == 0] = 255
                        fov_voxel_predict[fov_voxel_predict == 12] = 0
                        fov_voxel_label[fov_voxel_label == 0] = 255
                        fov_voxel_label[fov_voxel_label == 12] = 0
                        # fov_voxel_predict = fov_voxel_predict.cpu()
                        # fov_voxel_label = fov_voxel_label.cpu()

                        CalMeanIou_Fov.add_batch(fov_voxel_predict, fov_voxel_label)

                        if i and (i + 1) % 10 == 0:
                            frame_info = f'[{i} / {K_Frames}]'

                            status = CalMeanIou.get_stats(distributed=False)
                            sem_cls = status["iou_ssc"]
                            sem = status["iou_ssc_mean"]
                            geo = status["iou"]

                            fov_status = CalMeanIou_Fov.get_stats(distributed=False)
                            # fov_sem_cls = fov_status["iou_ssc"]
                            fov_sem = fov_status["iou_ssc_mean"]
                            fov_geo = fov_status["iou"]
                            # logger.info(f'Current fov iou of sem is {fov_sem_cls}')
                            logger.info(f'Current fov/glo {frame_info} iou of geo is {fov_geo:.5f} / {geo:.5f}')
                            logger.info(f'Current fov/glo {frame_info} iou of sem is {fov_sem:.5f} / {sem:.5f}')

                    if (i == K_Frames - 1):

                        label_mask = my_model.module.global_mask_thistime
                        assert len(label_mask) == 1
                        label_mask = label_mask[0].cpu()
                        voxel_predict = voxel_predict[label_mask]
                        voxel_label = voxel_label[label_mask]

                        CalMeanIou_Global.add_batch(voxel_predict, voxel_label)

                        assert len(my_model.module.global_gaussians) == 1, 'only support bs=1'

                if i_iter_val % print_freq == 0 and is_main_process():

                    global_step = f'{i_iter_val} / {len(val_dataset_loader)}'
                    global_status = CalMeanIou_Global.get_stats()
                    global_sem_cls = global_status["iou_ssc"]
                    global_sem = global_status["iou_ssc_mean"]
                    global_geo = global_status["iou"]
                    logger.info(f'Current global [{global_step}] iou of sem is {global_sem_cls}')
                    logger.info(f'Current global [{global_step}] iou of geo is {global_geo}')
                    logger.info(f'Current global [{global_step}] iou of sem is {global_sem}')

            global_status = CalMeanIou_Global.get_stats(distributed=distributed)
            global_sem_cls = global_status["iou_ssc"]
            global_sem = global_status["iou_ssc_mean"]
            global_geo = global_status["iou"]

            if is_main_process():
                logger.info(f'Final global iou of sem is {global_sem_cls}')
                logger.info(f'Final global iou of sem is {global_sem}')
                logger.info(f'Final global iou of geo is {global_geo}')
        return
    else:
        assert False

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/train_embodied_config.py')
    parser.add_argument('--model_config', default=None)
    parser.add_argument('--work-dir', type=str, default='/home/wyq/WorkSpace/workdir/train_embodied')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_freq', type=int, default=10)
    parser.add_argument('--save', action='store_true')

    args, _ = parser.parse_known_args()
    main(args)
