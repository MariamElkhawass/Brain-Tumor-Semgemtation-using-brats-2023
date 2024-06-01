import torch
import os
import argparse

from src import inference
from src.inference import main

parser = argparse.ArgumentParser(description='Brats validation and testing dataset inference')
parser.add_argument('--config', default='', type=str, metavar='PATH',
                    help='path(s) to the trained models config yaml you want to use', nargs="+")
parser.add_argument('--devices', required=False, type=str,default='0',
                    help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--on', default="val", choices=["val","train","test"])
parser.add_argument('--tta', action="store_true")
parser.add_argument('--seed', default=16111990)


if __name__ == '__main__':
    arguments = parser.parse_args()
    print(arguments.devices)
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
