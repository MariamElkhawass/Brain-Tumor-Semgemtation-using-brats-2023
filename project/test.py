import torch
import os
import argparse
import SimpleITK as sitk

#print(torch.cuda.current_device())
#print(torch.cuda.get_device_name(0))   

from src import train
from src.train import main

parser = argparse.ArgumentParser(description='Brats Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='Unet',
                    help='model architecture (default: Unet)')
parser.add_argument('--width', default=48, help='base number of features for Unet (x2 per downsampling)', type=int)
# DO not use data_aug argument this argument!!
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
# Warning: untested option!!
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint. Warning: untested option')
parser.add_argument('--devices', required=False, type=str, default='0', 
                    help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--debug', action="store_true")
parser.add_argument('--deep_sup', action="store_true")
parser.add_argument('--no_fp16', action="store_true")
parser.add_argument('--seed', default=16111990, help="seed for train/val split")
parser.add_argument('--warm', default=3, type=int, help="number of warming up epochs")

parser.add_argument('--val', default=3, type=int, help="how often to perform validation step")
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")
parser.add_argument('--norm_layer', default='group')
parser.add_argument('--swa', action="store_true", help="perform stochastic weight averaging at the end of the training")
parser.add_argument('--swa_repeat', type=int, default=5, help="how many warm restarts to perform")
parser.add_argument('--optim', choices=['adam', 'sgd', 'ranger', 'adamw'], default='ranger')
parser.add_argument('--com', help="add a comment to this run!")
parser.add_argument('--dropout', type=float, help="amount of dropout to use", default=0.)
parser.add_argument('--warm_restart', action='store_true', help='use scheduler warm restarts with period of 30')
parser.add_argument('--full', action='store_true', help='Fit the network on the full training set')

parser.add_argument('--lesion-wise', action='store_true', help='Perform lesion-wise training')

if __name__ == '__main__':
    arguments = parser.parse_args()
    print(arguments.devices)
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
