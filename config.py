import torch
import torch.nn as nn
import argparse
import datetime


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
        
parser = argparse.ArgumentParser(description='Training Option')        
 ### log setting
parser.add_argument('--log_file_name', type=str, default='./log/' + datetime.datetime.now().strftime('%Y-%m-%d') + '.log',
                    help='Log file name')
parser.add_argument('--logger_name', type=str, default='myLogger',
                    help='Logger name')

### device setting
parser.add_argument('--cpu', type=str2bool, default=False,
                    help='Use CPU to run code')
parser.add_argument('--num_gpu', type=int, default=1,
                    help='The number of GPU used in training')

### dataloader setting
parser.add_argument('--num_workers', type=int, default=1,
                    help='The number of workers when loading data')

### model setting
parser.add_argument('--num_res_blocks', type=str, default='16+16+8+4',
                    help='The number of residual blocks in each stage')
parser.add_argument('--n_feats', type=int, default=64,
                    help='The number of channels in network')
parser.add_argument('--res_scale', type=float, default=0.95,
                    help='Residual scale')
parser.add_argument('--SR_scale', type=int, default=2,
                    help='Scale of Supre-resolution')

### loss setting
parser.add_argument('--rec_w', type=float, default=1.,
                    help='The weight of reconstruction loss')
parser.add_argument('--struct_w', type=float, default=1,
                    help='The weight of adversarial loss')

### optimizer setting
parser.add_argument('--beta1', type=float, default=0.3,
                    help='The beta1 in Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.7,
                    help='The beta2 in Adam optimizer')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='The eps in Adam optimizer')
parser.add_argument('--lr_rate', type=float, default=2e-4,
                    help='Learning rate')

### training setting
parser.add_argument('--batch_size', type=int, default=2,
                    help='Training batch size')
parser.add_argument('--train_crop_size', type=int, default=40,
                    help='Training data crop size')
parser.add_argument('--num_epochs', type=int, default=50,
                    help='The number of training epochs')
parser.add_argument('--train_lr_path', type=str, default='G://train_toy/LRx2',
                    help='Low resolution dataset for training')
parser.add_argument('--train_hr_path', type=str, default='G://train_toy/HR',
                    help='High resolution dataset for training')
parser.add_argument('--train_ref_path', type=str, default='G://train_toy/Ref',
                    help='Ref dataset for training')


### validation setting
parser.add_argument('--val_lr_path', type=str, default='G://valid/LRx2',
                    help='Low resolution dataset for validation')
parser.add_argument('--val_hr_path', type=str, default='G://valid/HR',
                    help='High resolution dataset for validation')
parser.add_argument('--val_ref_path', type=str, default='G://valid/Ref',
                    help='Ref dataset for validation')
parser.add_argument('--val_freq', type=int, default=5,
                    help='Validation Frequency')


### test setting
parser.add_argument('--test_lr_path', type=str, default='G://test/LRx2',
                    help='Low resolution dataset for validation')
parser.add_argument('--test_hr_path', type=str, default='G://test/HR',
                    help='High resolution dataset for validation')
parser.add_argument('--test_ref_path', type=str, default='G://test/Ref',
                    help='Ref dataset for validation')

args = parser.parse_args()       

