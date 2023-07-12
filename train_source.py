import argparse
parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
parser.add_argument('-g', '--gpu', type=str, default='7', help='gpu id')
parser.add_argument('--resume', default=None, help='checkpoint path')
parser.add_argument(
    '--dataset', type=str, default='Domain3', help='test folder id contain images ROIs to test'
)
parser.add_argument(
    '--model', type=str, default='Deeplab', help='Deeplab'
)
parser.add_argument(
    '--batch-size', type=int, default=8, help='batch size for training the model'
)
parser.add_argument(
    '--group-num', type=int, default=1, help='group number for group normalization'
)
parser.add_argument(
    '--max-epoch', type=int, default=200, help='max epoch'
)
parser.add_argument(
    '--stop-epoch', type=int, default=200, help='stop epoch'
)
parser.add_argument(
    '--warmup-epoch', type=int, default=-1, help='warmup epoch begin train GAN'
)

parser.add_argument(
    '--interval-validate', type=int, default=5, help='interval epoch number to validate the model'
)
parser.add_argument(
    '--interval-save', type=int, default=10, help='interval epoch number to save the model'
)
parser.add_argument(
    '--lr', type=float, default=1e-3, help='learning rate',
)
parser.add_argument(
    '--lr-decrease-rate', type=float, default=0.98, help='ratio multiplied to initial lr',
)
parser.add_argument(
    '--lr-decrease-epoch', type=int, default=1, help='interval epoch number for lr decrease',
)
parser.add_argument(
    '--weight-decay', type=float, default=0, help='weight decay',
)
parser.add_argument(
    '--momentum', type=float, default=0.99, help='momentum',
)
parser.add_argument(
    '--data-dir',
    default='../Datasets/Fundus',
    help='data root path'
)
parser.add_argument(
    '--out-stride',
    type=int,
    default=16,
    help='out-stride of deeplabv3+',
)
parser.add_argument(
    '--sync-bn',
    type=bool,
    default=True,
    help='sync-bn in deeplabv3+',
)
parser.add_argument(
    '--freeze-bn',
    type=bool,
    default=False,
    help='freeze batch normalization of deeplabv3+',
)
parser.add_argument('--no-augmentation', action='store_true')

args = parser.parse_args()

from datetime import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import os.path as osp

# PyTorch includes
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import yaml
from train_process import Trainer

# Custom includes
from dataloaders import fundus_dataloader
from dataloaders import custom_transforms as trans
from networks.deeplabv3 import *


here = osp.dirname(osp.abspath(__file__))

def main():
    now = datetime.now()
    args.out = osp.join(here, 'logs_train', args.dataset, now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    multiply_gpu = False
    if (args.gpu).find(',') != -1:
        multiply_gpu = True
    cuda = torch.cuda.is_available()

    torch.manual_seed(42)
    if cuda:
        torch.cuda.manual_seed(42)

    # 1. dataset
    composed_transforms_tr = transforms.Compose([
        trans.RandomScaleCrop(512),
        trans.RandomRotate(),
        trans.RandomFlip(),
        trans.elastic_transform(),
        trans.add_salt_pepper_noise(),
        trans.adjust_light(),
        trans.eraser(),
        trans.Normalize_tf(),
        trans.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        # fundus_trans.RandomCrop(512),
        trans.Resize(512),
        trans.Normalize_tf(),  # this function separates (raw ground truth mask) into (2 masks)
        trans.ToTensor()
    ])

    if args.no_augmentation:
        domain = fundus_dataloader.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset,
                                                      split='train/ROIs', transform=composed_transforms_ts)
    else:
        domain = fundus_dataloader.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset,
                                                      split='train/ROIs', transform=composed_transforms_tr)
    domain_loader = DataLoader(domain, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    domain_val = fundus_dataloader.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset,
                                                      split='test/ROIs', transform=composed_transforms_ts)
    domain_loader_val = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                   pin_memory=True)

    # 2. model
    model = DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride,
                                sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    if cuda:
        model = model.cuda()

    start_epoch = 0
    start_iteration = 0

    # 3. optimizer
    if multiply_gpu:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_decrease_epoch, gamma=args.lr_decrease_rate)

    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)


        start_epoch = checkpoint['epoch'] + 1
        start_iteration = checkpoint['iteration'] + 1
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = Trainer.Trainer(
        cuda=cuda,
        multiply_gpu=multiply_gpu,
        model=model,
        optimizer=optim,
        scheduler=scheduler,
        lr=args.lr,
        val_loader=domain_loader_val,
        domain_loader=domain_loader,
        out=args.out,
        max_epoch=args.max_epoch,
        stop_epoch=args.stop_epoch,
        interval_validate=args.interval_validate,
        interval_save=args.interval_save,
        batch_size=args.batch_size,
        warmup_epoch=args.warmup_epoch
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()

if __name__ == '__main__':
    main()
