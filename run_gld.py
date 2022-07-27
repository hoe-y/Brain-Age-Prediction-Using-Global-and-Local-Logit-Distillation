import os
import torch
import random
import logging
import warnings
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from apex import amp
from torch.cuda.amp import autocast
from args import get_parser

from models.resnet_gld import resnet18, resnet50, resnet101, resnet152, resnet200

from torch.utils.data import DataLoader
from apex.parallel import DistributedDataParallel
from dataset.data import data_prefetcher, AllData
from utils.utils import *
from gld import *

def initialize():
    # get args
    args = get_parser()

    # warnings
    warnings.filterwarnings("ignore")

    # logger
    logger = logging.getLogger(__name__)

    # set seed
    seed = int(1111)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    # initialize logger
    logger.setLevel(level = logging.INFO)

    if not os.path.exists("logs"):
        os.makedirs("logs")

    handler = logging.FileHandler("logs/%s.txt" % args.env_name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return args, logger

def main():
    config, logger = initialize()
    config.nprocs = torch.cuda.device_count()
    main_worker(config, logger)

def main_worker(config, logger):
    model_names = ["resnet18", "resnet50", "resnet101", "resnet152", "resnet200"]
    models = [resnet18, resnet50, resnet101, resnet152, resnet200]

    best_acc1 = 99.0

    dist.init_process_group(backend='nccl')
    if not os.path.exists(config.save_root):
        try:
            os.makedirs(config.save_root)
        except:
            pass # multiple processors bug
    # create model
    model = models[model_names.index(config.arch)](output_dim=88)
    T_model = None

    torch.cuda.set_device(config.local_rank)
    model.cuda()

    # load teacher model checkpoint
    if config.teacher_path is not None:
        T_model = models[model_names.index(config.t_arch)](output_dim=88)
        
        checkpoint = torch.load(config.teacher_path) 
        T_model.load_state_dict(checkpoint, strict=False)
        #T_model = DistributedDataParallel(T_model).to(torch.device("cuda"))
        
    # print pre-trained teacher and to-be-trained student information
        t_num_parameters = round((sum(l.nelement() for l in T_model.parameters()) / 1e+6), 3)
        s_num_parameters = round((sum(l.nelement() for l in model.parameters()) / 1e+6), 3)
        print("teacher name : ", config.t_arch)
        print("teacher parameters : ", t_num_parameters, "M")
        print("student name : ", config.arch)
        print("student parameters : ", s_num_parameters, "M")
        log = open(config.save_root+'/log.txt', 'a')
        log.write("teacher name : {}\n".format(config.t_arch))
        log.write("teacher parameters : {} M\n".format(t_num_parameters))
        log.write("student name : {}\n".format(config.arch))
        log.write("student parameters : {} M\n".format(s_num_parameters))
        log.close()

    config.batch_size = int(config.batch_size / config.nprocs)
    optimizer = torch.optim.Adam(model.parameters(),lr = config.lr,weight_decay = 0.00005)
    model, optimizer = amp.initialize(model, optimizer, opt_level=config.opt_level)
    model = DistributedDataParallel(model).to(torch.device("cuda"))
    #model = torch.nn.DataParallel(model).to(torch.device("cuda"))
    cudnn.benchmark = True

    # Data loading code
    train_data = AllData(config.data, train = True)
    val_data = AllData(config.data, train = False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)

    train_loader = DataLoader(train_data,config.batch_size,
                        shuffle=False,num_workers=8,pin_memory = True, sampler = train_sampler)
    val_loader = DataLoader(val_data,config.batch_size,
                        shuffle=False,num_workers=4,pin_memory = True, sampler = val_sampler)

    distill_criterion = GLDLoss(alpha=config.gld_alpha, beta=config.gld_beta, spatial_size=24, div=config.gld_div)

    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, config)
        #train(train_loader, model, T_model, optimizer, epoch, config, logger)
        if T_model is not None:
            tr_mae, tr_fc_loss, tr_d_loss = gld_distillation(train_loader, T_model, model, distill_criterion, optimizer, epoch, config, logger)    
        else:
            tr_mae, tr_loss = gld_distillation(train_loader, T_model, model, distill_criterion, optimizer, epoch, config, logger)    

        mae = validate(val_loader, model, config, logger)
        #mae, te_loss = validate(val_loader, model, config, logger)

        is_best = mae < best_acc1
        best_acc1 = min(mae, best_acc1)

        if is_best and config.local_rank == 0:
            if T_model is not None:
                state = {
                    'epoch': epoch + 1,
                    'train_fc_loss': tr_fc_loss, 
                    'train_d_loss': tr_d_loss,
                    'train_mae': tr_mae,
                    'test_mae': mae,
                    }
            else:
                state = {
                    'epoch': epoch + 1, 
                    'train_fc_loss': tr_loss, 
                    'train_mae': tr_mae,
                    'test_mae': mae,
                    'amp': amp.state_dict(),
                    }
            torch.save(state, config.save_root+'/_%s_epoch_%s-----%s.pt' % ( epoch, best_acc1, config.env_name))
            log = open(config.save_root+'/log.txt', 'a')
            log.write("[{}K] {:.2f} (Best MAE: {:.2f} )\n"
                .format(epoch, mae, best_acc1))
            log.close()

def gld_distillation(train_loader, teacher, student, criterion, optimizer, epoch, config, logger):
    losses = AverageMeter('Loss', ':.4e')
    loss_mae = AverageMeter('mae1', ':6.2f')
    ce_losses = AverageMeter('task_loss', ':.4e')
    dis_losses = AverageMeter('distill_loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses, loss_mae],
                             prefix="Epoch: [{}]".format(epoch), logger = logger)

    # switch to train mode
    student.train()
    if teacher is not None:
        teacher.eval()

    prefetcher = data_prefetcher(train_loader)
    images, target, yy, bc, indices = prefetcher.next()

    i = 0
    optimizer.zero_grad()
    optimizer.step()

    while images is not None:
        
        s_fr, s_output = student(images)

        if teacher is not None:

            teacher.half().cuda()

            t_fr, t_output = teacher(images)

            S_pred,T_pred = torch.sum(torch.exp(s_output) * bc, dim = 1),torch.sum(torch.exp(t_output) * bc, dim = 1)
           
            teacher2 = nn.DataParallel(teacher)
            s_f1 = student.module.f1
            s_f2 = student.module.f2

            t_f1 = teacher2.module.f1
            t_f2 = teacher2.module.f2

            # distilling
            task_loss, distill_loss = criterion(t_fr, s_fr, t_f1, t_f2, s_f1, s_f2, yy)
            loss = task_loss + distill_loss
        else:
            S_pred = torch.sum(torch.exp(s_output) * bc, dim = 1)
            loss = my_KLDivLoss(s_output, yy).cuda()

        # measure accuracy and record loss
        mae = torch.nn.L1Loss()(S_pred, target)

        torch.distributed.barrier() 

        reduced_loss = reduce_mean(loss, config.nprocs)
        reduced_mae = reduce_mean(mae, config.nprocs)

        losses.update(reduced_loss.item(), images.size(0))
        loss_mae.update(reduced_mae.item(), images.size(0))
        
        if teacher is not None:
            ce_losses.update(task_loss.item(), images.size(0))
            dis_losses.update(distill_loss.item(), images.size(0))

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        if i % config.print_freq == 0:
            progress.display(i)

        i += 1

        images, target, yy, bc, indices = prefetcher.next()

    logger.info("[train mae]: %.4f" % float(loss_mae.avg))

    if teacher is not None:
        return loss_mae.avg, ce_losses, dis_losses
    else:
        return loss_mae.avg, losses

def validate(val_loader, model, config, logger):

    loss_mae = AverageMeter('mae1', ':6.2f')

    progress = ProgressMeter(len(val_loader), [loss_mae], prefix='Test: ', logger = logger)
    model.eval()

    with torch.no_grad():
        prefetcher = data_prefetcher(val_loader)
        images, target, yy, bc, _ = prefetcher.next()
        i = 0
        while images is not None:

            _, out = model(images)

            prob = torch.exp(out)
            pred = torch.sum(prob * bc, dim = 1)
            mae = torch.nn.L1Loss()(pred, target) 

            torch.distributed.barrier()
            reduced_mae = reduce_mean(mae, config.nprocs)

            loss_mae.update(reduced_mae.item(), images.size(0))


            if i % config.print_freq == 0:
                progress.display(i)

            i += 1

            images, target, _, bc, _ = prefetcher.next()

        logger.info("[val mae]: %.4f" % float(loss_mae.avg))
    return loss_mae.avg


if __name__ == '__main__':
    main()
