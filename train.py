import argparse
import os
import torch
import torch.nn.functional as F
import datetime
from model.DBS_group_prompt import Spider_ConvNeXt,Spider_Swin
from utils.dataset_rgb_strategy2 import SalObjDataset
from utils.utils import adjust_lr, AvgMeter
import torch.nn as nn
from torch.cuda import amp
import torch.distributed as dist
from contextlib import contextmanager
import torch.utils.data as data


def get_loader(image_root, gt_root, batchsize, trainsize):
    dataset = SalObjDataset(image_root, gt_root, trainsize)
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  num_workers=12,
                                  pin_memory=True,
                                  sampler=sampler)
    return data_loader, sampler


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_master():
    return get_rank() == 0


@contextmanager
def torch_distributed_zero_first(rank: int):
    """Decorator to make all processes in distributed training wait for each local_master to do something."""
    if is_dist_avail_and_initialized() and rank not in [-1, 0]:
        torch.distributed.barrier()
    # 这里的用法其实就是协程的一种哦。
    yield
    if is_dist_avail_and_initialized() and rank == 0:
        torch.distributed.barrier()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr_gen', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
    parser.add_argument('-beta1_gen', type=float, default=0.5, help='beta of Adam for generator')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
    parser.add_argument('--feat_channel', type=int, default=64, help='reduced channel of saliency feat')
    return parser.parse_args()
        

def train():
    opt = get_args()
    print('Generator Learning Rate: {}'.format(opt.lr_gen))

    print('分布式开始初始化...')
    distributed = int(os.environ["WORLD_SIZE"]) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", world_size=world_size, init_method="env://")
    print('分布式初始化完成!')
    is_master = (distributed and (local_rank == 0)) or (not distributed)
    
    ## load data
    image_sod_root = "/root/autodl-tmp/datasets/dbs/data_path/DUTS-TR_img.txt"
    image_cod_root = "/root/autodl-tmp/datasets/dbs/data_path/COD_train_img.txt"
    image_shadow_root = "/root/autodl-tmp/datasets/dbs/data_path/Shadow_img.txt"
    image_transparent_root = "/root/autodl-tmp/datasets/dbs/data_path/transparent_img.txt"
    image_polyp_root = "/root/autodl-tmp/datasets/dbs/data_path/Polyp_train_img.txt"   
    image_covid_root = "/root/autodl-tmp/datasets/dbs/data_path/COVID-19_img.txt"
    image_breast_root = "/root/autodl-tmp/datasets/dbs/data_path/breast_train_img.txt"
    image_skin_root = "/root/autodl-tmp/datasets/dbs/data_path/skin_img.txt"   

    gt_sod_root = "/root/autodl-tmp/datasets/dbs/data_path/DUTS-TR_gt.txt"
    gt_cod_root = "/root/autodl-tmp/datasets/dbs/data_path/COD_train_gt.txt"
    gt_shadow_root = "/root/autodl-tmp/datasets/dbs/data_path/Shadow_gt.txt"
    gt_transparent_root = "/root/autodl-tmp/datasets/dbs/data_path/transparent_gt.txt"  
    gt_polyp_root = "/root/autodl-tmp/datasets/dbs/data_path/Polyp_train_gt.txt"  
    gt_covid_root = "/root/autodl-tmp/datasets/dbs/data_path/COVID-19_gt.txt"
    gt_breast_root = "/root/autodl-tmp/datasets/dbs/data_path/breast_train_gt.txt"
    gt_skin_root = "/root/autodl-tmp/datasets/dbs/data_path/skin_gt.txt"  


    train_sod_loader, train_sod_sampler = get_loader(image_sod_root, gt_sod_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    train_cod_loader, train_cod_sampler = get_loader(image_cod_root, gt_cod_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    train_shadow_loader, train_shadow_sampler = get_loader(image_shadow_root, gt_shadow_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    train_transparent_loader, train_transparent_sampler = get_loader(image_transparent_root, gt_transparent_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    train_polyp_loader, train_polyp_sampler = get_loader(image_polyp_root, gt_polyp_root, batchsize=opt.batchsize, trainsize=opt.trainsize)   
    train_covid_loader, train_covid_sampler = get_loader(image_covid_root, gt_covid_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    train_breast_loader, train_breast_sampler = get_loader(image_breast_root, gt_breast_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    train_skin_loader, train_skin_sampler = get_loader(image_skin_root, gt_skin_root, batchsize=opt.batchsize, trainsize=opt.trainsize)  
    total_step = len(train_sod_loader)

    size_rates = [1]  # multi-scale training
    use_fp16 = True
        
    save_path = './saved_model/Spider_Swin_model'
    with torch_distributed_zero_first(rank=local_rank):
        os.makedirs(save_path, exist_ok=True)
        
    log_path = os.path.join(save_path,str(datetime.datetime.now()) + '.txt')
    if is_master:
        open(log_path, 'w')
    
    print("开始初始化模型，优化器...")
    generator = Spider_Swin()
    generator.cuda()
    
    generator_optimizer = torch.optim.Adam(generator.parameters(), opt.lr_gen)
    scaler = amp.GradScaler(enabled=use_fp16)
    
    generator = nn.parallel.DistributedDataParallel(
        generator,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )
    
    
    print("Start Training...")
    for epoch in range(1, opt.epoch + 1):
        train_sod_sampler.set_epoch(epoch)
        train_cod_sampler.set_epoch(epoch)
        train_shadow_sampler.set_epoch(epoch)
        train_transparent_sampler.set_epoch(epoch)
        train_polyp_sampler.set_epoch(epoch)
        train_covid_sampler.set_epoch(epoch)
        train_breast_sampler.set_epoch(epoch)
        train_skin_sampler.set_epoch(epoch)
        generator.train()
        loss_record = AvgMeter()
        print('Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))

        for i, (
                (image_sod, gt_sod),
                (image_cod, gt_cod),
                (image_shadow, gt_shadow),
                (image_transparent, gt_transparent),
                (image_polyp, gt_polyp),
                (image_covid, gt_covid),
                (image_breast, gt_breast),
                (image_skin, gt_skin),
        ) in enumerate(zip(
            train_sod_loader,
            train_cod_loader,
            train_shadow_loader,
            train_transparent_loader,
            train_polyp_loader,
            train_covid_loader,
            train_breast_loader,
            train_skin_loader,
        ), start=1):
            # 4,6,3,H,W
            images = torch.stack([image_sod, image_cod, image_shadow, image_transparent, image_polyp, image_covid, image_breast, image_skin], dim=1)
            gts = torch.stack([gt_sod, gt_cod, gt_shadow, gt_transparent, gt_polyp, gt_covid, gt_breast, gt_skin], dim=1)
            images = images.cuda()
            gts = gts.cuda()
            recurrent_set = 4
            batch_set = gt_sod.shape[0] // recurrent_set
            for rate in size_rates:
                for curr_idx in range(recurrent_set):
                    # 6,3,H,W
                    query_image = images[curr_idx*batch_set:(curr_idx+1)*batch_set]
                    query_gt = gts[curr_idx*batch_set:(curr_idx+1)*batch_set]

                    # 3,6,3,H,W, skip the curr_idx-th sample
                    support_images = torch.cat([images[:curr_idx*batch_set], images[(curr_idx+1)*batch_set:]], dim=0)
                    support_gts = torch.cat([gts[:curr_idx*batch_set], gts[(curr_idx+1)*batch_set:]], dim=0)

                    # multi-scale training samples
                    trainsize = int(round(opt.trainsize * rate / 32) * 32)
                    if rate != 1:
                        query_image = F.upsample(query_image, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                        query_gt = F.upsample(query_gt, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                    with amp.autocast(enabled=use_fp16):
                        support_images = support_images.permute(1, 0, 2, 3, 4) # => 6,3,3,H,W
                        support_gts = support_gts.permute(1, 0, 2, 3, 4) # => 6,3,3,H,W
                        
                        query_image = query_image.permute(1, 0, 2, 3, 4)
                        query_image = query_image.reshape(-1,3,image_sod.shape[2],image_sod.shape[3])
                        query_gt = query_gt.permute(1, 0, 2, 3, 4)
                        query_gt = query_gt.reshape(-1,1,image_sod.shape[2],image_sod.shape[3])
                        
                        # print(query_image.shape,support_images[0].shape)
                        
                        
                        output_fpn = generator(query_image, support_images,support_gts)  # 6,6,H,W
                                   
                        loss1 = structure_loss(output_fpn[0:batch_set, 0:1], query_gt[0:batch_set])
                        loss2 = structure_loss(output_fpn[batch_set:batch_set*2, 1:2], query_gt[batch_set:batch_set*2])
                        loss3 = structure_loss(output_fpn[batch_set*2:batch_set*3, 2:3], query_gt[batch_set*2:batch_set*3])
                        loss4 = structure_loss(output_fpn[batch_set*3:batch_set*4, 3:4], query_gt[batch_set*3:batch_set*4])
                        loss5 = structure_loss(output_fpn[batch_set*4:batch_set*5, 4:5], query_gt[batch_set*4:batch_set*5])
                        loss6 = structure_loss(output_fpn[batch_set*5:batch_set*6, 5:6], query_gt[batch_set*5:batch_set*6])
                        loss7 = structure_loss(output_fpn[batch_set*6:batch_set*7, 6:7], query_gt[batch_set*6:batch_set*7])
                        loss8 = structure_loss(output_fpn[batch_set*7:batch_set*8, 7:8], query_gt[batch_set*7:batch_set*8])
                        loss = 1*(loss1 + loss2  + loss3 + loss4  + loss5  + loss6 + loss7 + loss8)

                    generator_optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(generator_optimizer)
                    scaler.update()

                if rate == 1:
                    loss_record.update(loss.data, opt.batchsize)
                    
            if is_master:
                if i % 10 == 0 or i == total_step:
                    print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}'.
                          format(datetime.datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))

                    log = ('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}'.
                          format(datetime.datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))
                    open(log_path, 'a').write(log + '\n')

        adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

        if is_master:
            if epoch % 5 == 0:
                torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
            if epoch % opt.epoch == 0:
                torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')

if __name__ == '__main__':
    # freeze_support()
    train()
    # os.system("/usr/bin/shutdown")