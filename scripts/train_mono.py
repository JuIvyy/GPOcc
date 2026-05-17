import os, time, argparse, os.path as osp, numpy as np
import torch
import gc
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from gpocc.utils.iou_eval import IOUEvalBatch
from gpocc.utils.iou_as_iso import SSCMetrics, SSCMetricsTorch
from gpocc.utils.loss_record import LossRecord
from gpocc.utils.load_save_util import revise_ckpt, revise_ckpt_2
from gpocc.loss.depth_loss import DepthLoss

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim.optimizer.builder import build_optim_wrapper
from mmengine.logging.logger import MMLogger
from mmengine.utils import symlink
from timm.scheduler import CosineLRScheduler
import open3d as o3d
import cv2
from einops import einsum
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
import pickle

import warnings
warnings.filterwarnings("ignore")
from PIL import Image


def pass_print(*args, **kwargs):
    pass


def is_main_process():
    if not dist.is_available():
        return True
    elif not dist.is_initialized():
        return True
    else:
        return dist.get_rank() == 0


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def save_model(model, optimizer=None, scheduler=None, epoch=None, **kwargs):
    dict_to_save = {
        'state_dict': model.state_dict()
    }
    if optimizer is not None:
        dict_to_save['optimizer'] = optimizer.state_dict()

    if scheduler is not None:
        dict_to_save['scheduler'] = scheduler.state_dict()

    if len(kwargs):
        dict_to_save.update(kwargs)

    if epoch is not None:
        dict_to_save['epoch'] = epoch
        save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch}.pth')
    else:
        save_file_name = os.path.join(os.path.abspath(args.work_dir), f'ckpt.pth')

    torch.save(dict_to_save, save_file_name)
    dst_file = osp.join(args.work_dir, 'latest.pth')
    symlink(save_file_name, dst_file)


def main(args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
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

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend="nccl",
        # init_method=f"env://", 
        world_size=world_size,
    )
    rank, world_size = get_dist_info()

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
    my_model = build_model(cfg.model).cuda()
    
    if cfg.flag_depthanything_as_gt:
        my_model.depthanything.requires_grad_(False)

    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')

    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        if cfg.get('track_running_stats', False):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model,
            # device_ids=[gpu],
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)

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

    # get optimizer, loss, scheduler
    amp = cfg.get('amp', True)
    optimizer = build_optim_wrapper(my_model, cfg.optimizer_wrapper)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    from gpocc.loss import GPD_LOSS
    loss_func = GPD_LOSS.build(cfg.loss).cuda()
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=len(train_dataset_loader)*max_num_epochs,
        lr_min=1e-6,
        warmup_t=1000, # FIXME
        warmup_lr_init=1e-6,
        t_in_epochs=False
    )

    # CalMeanIou = SSCMetrics(n_classes=12)
    CalMeanIou = SSCMetricsTorch(n_classes=12)
    # resume and load
    epoch = 0
    best_val_iou = 0
    best_val_miou = 0
    global_iter = 0

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from

    logger.info(f'resume from: {cfg.resume_from}')
    logger.info(f'work dir: {args.work_dir}')

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(my_model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
        if not args.evaluate:
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        if 'best_val_iou' in ckpt:
            best_val_iou = ckpt['best_val_iou']
        if 'best_val_miou' in ckpt:
            best_val_miou = ckpt['best_val_miou']
        global_iter = ckpt['global_iter']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        state_dict = revise_ckpt(state_dict)
        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:
            state_dict = revise_ckpt_2(state_dict)
            print(my_model.load_state_dict(state_dict, strict=False))

    metas_tensor_keys_inv = ['depth_gt_np_valid', 'depth_gt_np', 'name', 'cam2img', 'world2img', 'rgb_path', 'depth_path','num_depth', 'occ_mask_valid', 'occ_mask_valid_fov', 'img_shape', 'img_aug_matrix']

    total_params = sum(p.numel() for p in my_model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f} M")

    if args.evaluate:
        my_model.eval()
        CalMeanIou.reset()
        # loss_record = LossRecord(loss_func=loss_func)
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        num_gaussians = []
        with torch.no_grad():
            for i_iter_val, data in enumerate(tqdm(val_dataset_loader)):

                for i in range(len(data)):
                    if isinstance(data[i], torch.Tensor):
                        data[i] = data[i].cuda()
                (imgs, metas, label) = data

                for k, v in metas[0].items():
                    if not (k in metas_tensor_keys_inv):
                        metas[0][k] = torch.tensor(v).cuda()
                metas[0]['img_depthbranch'] = metas[0]['img_depthbranch'].cuda()

                with torch.cuda.amp.autocast(enabled=amp):

                    result_dict, my_occ, predtoreturn = my_model(
                        imgs=imgs,
                        metas=metas,
                        points=None,
                        label=label,
                        grad_frames=None,
                        test_mode=True
                    )

                voxel_predict = result_dict['ce_input'].argmax(dim=1).long()
                voxel_label = result_dict['ce_label'].long()

                voxel_predict[voxel_predict == 0] = 255
                voxel_predict[voxel_predict == 12] = 0
                voxel_label[voxel_label == 0] = 255
                voxel_label[voxel_label == 12] = 0

                if args.eval_fov:
                    fov_mask = metas[0]['fov_mask']
                    voxel_label[0][fov_mask == 0] = 255
                    voxel_predict[0][fov_mask == 0] = 255

                CalMeanIou.add_batch(voxel_predict, voxel_label)

                if i_iter_val % print_freq == 0:
                    stats = CalMeanIou.get_stats(distributed=False)
                    info_sem_cls = stats["iou_ssc"]
                    info_sem = stats["iou_ssc_mean"]
                    info_geo = stats["iou"]

                    if is_main_process():
                        logger.info(f'Current val iou of sem_cls is {info_sem_cls}')
                        logger.info(f'Current val iou of sem is {info_sem}')
                        logger.info(f'Current val iou of geo is {info_geo}')

                gc.collect()
                torch.cuda.empty_cache()

            stats = CalMeanIou.get_stats(distributed=distributed)

            info_sem_cls = stats["iou_ssc"]
            info_sem = stats["iou_ssc_mean"]
            info_geo = stats["iou"]

            if is_main_process():
                logger.info(f'Current val iou of sem_cls is {info_sem_cls}')
                logger.info(f'Current val iou of sem is {info_sem}')
                logger.info(f'Current val iou of geo is {info_geo}')

        return


    # training
    while epoch < max_num_epochs:

        my_model.train()
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_record = LossRecord(loss_func=loss_func)
        time.sleep(1)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, data in enumerate(train_dataset_loader):
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].cuda()
            (imgs, metas, label) = data

            for k, v in metas[0].items():
                if not (k in metas_tensor_keys_inv):
                    metas[0][k] = torch.tensor(v).cuda()
            metas[0]['img_depthbranch'] = metas[0]['img_depthbranch'].cuda()
            # forward + backward + optimize
            data_time_e = time.time()

            with torch.cuda.amp.autocast(enabled=amp):
                result_dict, sparse_result_dict, predtoreturn = my_model(
                    imgs=imgs, metas=metas,
                    points=None, label=label,
                    grad_frames=cfg.grad_frames,
                    test_mode=False)

            assert cfg.ignore_label == 0

            # TODO hard code
            if 'vggt' in args.py_config.lower() or 'dpt' in args.py_config.lower():
                total_loss = 0.
                all_loss_dict = {}

                # if getattr(cfg, 'no_depth_loss', False):
                depth_loss = DepthLoss(
                    input_dict=dict(
                        depth_preds='depth_pred',
                        depth_labels='depth_gt',
                    ),
                    weight=cfg.get('depth_loss_weight', 2.0),
                )(result_dict)

                total_loss = total_loss + depth_loss
                all_loss_dict['depth_loss'] = depth_loss.detach().item()

                if 'ce_label' in result_dict:
                    loss, loss_dict = loss_func(result_dict)
                    all_loss_dict.update(loss_dict)
                    total_loss = total_loss + loss

                if sparse_result_dict is not None:
                    if isinstance(sparse_result_dict, dict) and 'ce_label' in sparse_result_dict:
                        sparse_loss, sparse_loss_dict = loss_func(sparse_result_dict)
                        for k, v in sparse_loss_dict.items():
                            all_loss_dict[f'sparse_{k}'] = v
                    else:
                        sparse_loss = sparse_result_dict
                    total_loss = total_loss + sparse_loss

                loss_record.update(loss=total_loss.item(), loss_dict=all_loss_dict)
            else:
                total_loss, loss_dict = loss_func(result_dict)
                loss_record.update(loss=total_loss.item(), loss_dict=loss_dict)

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)

            valid_grad = True

            scaler.step(optimizer)
            scaler.update()
            scheduler.step_update(global_iter)
            time_e = time.time()

            global_iter += 1
            if i_iter % print_freq == 0 and is_main_process():
                lr = optimizer.param_groups[0]['lr']
                loss_info = loss_record.loss_info()
                logger.info('[TRAIN] Epoch %d Iter %5d/%d   ' % (epoch+1, i_iter, len(train_dataset_loader)) + loss_info +
                            'GradNorm: %.3f,   lr: %.7f,   time: %.3f (%.3f)' % (grad_norm, lr, time_e - time_s, data_time_e - data_time_s))
                loss_record.reset()
            data_time_s = time.time()
            time_s = time.time()
            
        # save checkpoint
        if is_main_process():
            save_model(
                my_model, optimizer,
                scheduler, epoch + 1, 
                global_iter=global_iter,
                best_val_iou=best_val_iou,
                best_val_miou=best_val_miou)

        epoch += 1

        # eval
        if epoch % eval_freq == 0:
            my_model.eval()
            CalMeanIou.reset()
            loss_record = LossRecord(loss_func=loss_func)
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            with torch.no_grad():
                for i_iter_val, data in enumerate(val_dataset_loader):
                    for i in range(len(data)):
                        if isinstance(data[i], torch.Tensor):
                            data[i] = data[i].cuda()
                    (imgs, metas, label) = data
                    
                    for k, v in metas[0].items():
                        if not (k in metas_tensor_keys_inv):
                            metas[0][k] = torch.tensor(v).cuda()
                    metas[0]['img_depthbranch'] = metas[0]['img_depthbranch'].cuda()

                    with torch.cuda.amp.autocast(enabled=amp):
                        result_dict, my_occ, predtoreturn = my_model(
                            imgs=imgs,
                            metas=metas,
                            points=None,
                            label=label,
                            grad_frames=None,
                            test_mode=True
                        )

                    # loss, loss_dict = loss_func(result_dict)
                    # loss_record.update(loss=loss.item(), loss_dict=loss_dict)

                    voxel_predict = result_dict['ce_input'].argmax(dim=1).long() # [1, 60, 60, 36]
                    voxel_label = result_dict['ce_label'].long() # [1, 60, 60, 36]

                    voxel_predict[voxel_predict == 0] = 255
                    voxel_predict[voxel_predict == 12] = 0
                    voxel_label[voxel_label == 0] = 255
                    voxel_label[voxel_label == 12] = 0

                    CalMeanIou.add_batch(voxel_predict, voxel_label)
                                        
                    gc.collect()
                    torch.cuda.empty_cache()

            stats = CalMeanIou.get_stats(distributed=distributed)

            info_sem_cls = stats["iou_ssc"]
            info_sem = stats["iou_ssc_mean"]
            info_geo = stats["iou"]

            logger.info(f'Current val iou of sem_cls is {info_sem_cls}')
            logger.info(f'Current val iou of sem is {info_sem}')
            logger.info(f'Current val iou of geo is {info_geo}')
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/train_mono_config.py')
    parser.add_argument('--work-dir', type=str, default='/home/wyq/WorkSpace/workdir/train_mono')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_freq', type=int, default=10)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--eval_fov', action='store_true')

    args, _ = parser.parse_known_args()
    main(args)
