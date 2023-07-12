from datetime import datetime
import os
import os.path as osp
import timeit
from torchvision.utils import make_grid
import time

import numpy as np
import pytz
import torch
import torch.nn.functional as F

from tensorboardX import SummaryWriter

import tqdm
import socket
from utils.metrics import *
from utils.Utils import *

bceloss = torch.nn.BCELoss()
mseloss = torch.nn.MSELoss()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Trainer(object):

    def __init__(self, cuda, multiply_gpu, model, optimizer, scheduler, val_loader, domain_loader, out, max_epoch, stop_epoch=None,
                 lr=1e-3, interval_validate=10, interval_save=10, batch_size=8, warmup_epoch=10):
        self.cuda = cuda
        self.multiply_gpu = multiply_gpu
        self.warmup_epoch = warmup_epoch
        self.model = model
        self.optim = optimizer
        self.scheduler = scheduler
        self.lr = lr
        # self.lr_decrease_rate = lr_decrease_rate
        # self.lr_decrease_epoch = lr_decrease_epoch
        self.batch_size = batch_size

        self.val_loader = val_loader
        self.domain_loader = domain_loader
        self.time_zone = 'Asia/Shanghai'
        self.timestamp_start = datetime.now(pytz.timezone(self.time_zone))

        self.interval_validate = interval_validate
        self.interval_save = interval_save

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss_seg',
            'valid/loss_CE',
            'valid/cup_dice',
            'valid/disc_dice',
            'elapsed_time',
            'best_epoch'
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        log_dir = os.path.join(self.out, 'tensorboard',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch
        self.best_mean_dice = 0.0
        self.best_epoch = -1


    def validate_fundus(self):
        training = self.model.training
        self.model.eval()

        val_loss = 0.0
        val_fundus_dice = {'cup': 0.0, 'disc': 0.0}
        data_num_cnt = 0.0
        metrics = []
        with torch.no_grad():
            for batch_idx, sample in tqdm.tqdm(
                    enumerate(self.val_loader), total=len(self.val_loader),
                    desc='Valid iteration=%d' % self.iteration, ncols=80,
                    leave=False):
                data = sample['image']
                target_map = sample['label']
                if self.cuda:
                    data, target_map = data.cuda(), target_map.cuda()
                with torch.no_grad():
                    predictions, _ = self.model(data)

                loss = F.binary_cross_entropy_with_logits(predictions, target_map)
                loss_data = loss.data.item()
                if np.isnan(loss_data):
                    raise ValueError('loss is nan while validating')
                val_loss += loss_data

                dice_cup, dice_disc = dice_coeff_2label(predictions, target_map)
                val_fundus_dice['cup'] += np.sum(dice_cup)
                val_fundus_dice['disc'] += np.sum(dice_disc)
                data_num_cnt += float(dice_cup.shape[0])

            val_loss /= data_num_cnt
            val_fundus_dice['cup'] /= data_num_cnt
            val_fundus_dice['disc'] /= data_num_cnt
            metrics.append((val_loss, val_fundus_dice['cup'], val_fundus_dice['disc']))

            self.writer.add_scalar('val_data/loss_CE', val_loss, self.epoch * (len(self.domain_loader)))
            self.writer.add_scalar('val_data/val_CUP_dice', val_fundus_dice['cup'], self.epoch * (len(self.domain_loader)))
            self.writer.add_scalar('val_data/val_DISC_dice', val_fundus_dice['disc'], self.epoch * (len(self.domain_loader)))

            mean_dice = (val_fundus_dice['cup'] + val_fundus_dice['disc'])/2
            is_best = mean_dice > self.best_mean_dice
            if is_best:
                self.best_epoch = self.epoch + 1
                self.best_mean_dice = mean_dice

                torch.save({
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.model.__class__.__name__,
                    'optim_state_dict': self.optim.state_dict(),
                    'model_state_dict': self.model.module.state_dict() if self.multiply_gpu else self.model.state_dict(),
                    'learning_rate_gen': get_lr(self.optim),
                    'best_mean_dice': self.best_mean_dice,
                }, osp.join(self.out, 'checkpoint_%d.pth.tar' % self.best_epoch))
            else:
                if (self.epoch + 1) % self.interval_save == 0:
                    torch.save({
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'arch': self.model.__class__.__name__,
                        'optim_state_dict': self.optim.state_dict(),
                        'model_state_dict': self.model.module.state_dict() if self.multiply_gpu else self.model.state_dict(),
                        'learning_rate_gen': get_lr(self.optim),
                        'best_mean_dice': self.best_mean_dice,
                    }, osp.join(self.out, 'checkpoint_%d.pth.tar' % (self.epoch + 1)))

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.now(pytz.timezone(self.time_zone)) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [''] + list(metrics) + [elapsed_time] + [self.best_epoch]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            self.writer.add_scalar('best_model_epoch', self.best_epoch, self.epoch * (len(self.domain_loader)))
            if training:
                self.model.train()


    def train_epoch(self):
        self.model.train()
        self.running_seg_loss = 0.0

        start_time = timeit.default_timer()
        for batch_idx, sample in tqdm.tqdm(
                enumerate(self.domain_loader), total=len(self.domain_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            iteration = batch_idx + self.epoch * len(self.domain_loader)
            self.iteration = iteration

            assert self.model.training

            self.optim.zero_grad()

            # train
            for param in self.model.parameters():
                param.requires_grad = True

            image = sample['image'].cuda()
            target_map = sample['label'].cuda()

            pred, _ = self.model(image)
            pred = torch.sigmoid(pred)
            loss_seg = bceloss(pred, target_map)

            self.running_seg_loss += loss_seg.item()
            self.running_seg_loss /= len(self.domain_loader)

            loss_seg_data = loss_seg.data.item()
            if np.isnan(loss_seg_data):
                raise ValueError('loss is nan while training')

            loss_seg.backward()
            self.optim.step()

            # write image log
            if iteration % 30 == 0:
                grid_image = make_grid(image[0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('Domain/image', grid_image, iteration)
                grid_image = make_grid(target_map[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('Domain/target_cup', grid_image, iteration)
                grid_image = make_grid(target_map[0, 1, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('Domain/target_disc', grid_image, iteration)
                grid_image = make_grid(pred[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('Domain/prediction_cup', grid_image, iteration)
                grid_image = make_grid(pred[0, 1, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('Domain/prediction_disc', grid_image, iteration)


            self.writer.add_scalar('train/loss_seg', loss_seg_data, iteration)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.now(pytz.timezone(self.time_zone)) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss_seg_data] + [''] * 3 + [elapsed_time] + ['']
                log = map(str, log)
                f.write(','.join(log) + '\n')

        stop_time = timeit.default_timer()

        print('\n[Epoch: %d] lr:%f,  Average segLoss: %f, Execution time: %.5f\n' %
              (self.epoch, get_lr(self.optim), self.running_seg_loss, stop_time - start_time))


    def train(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break

            self.scheduler.step()
            self.writer.add_scalar('lr', get_lr(self.optim), self.epoch * (len(self.domain_loader)))

            if (self.epoch+1) % self.interval_validate == 0:
                self.validate_fundus()
        self.writer.close()



