import argparse
import os

import torch
import torch.nn.parallel
import torch.optim

from Models import modelpool
from Preprocess import datapool
from utils import val, seed_all

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('-b', '--batch_size', default=200, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=13, type=int, choices=[13, 42, 2026],
                    help='seed for initializing training. ')
parser.add_argument('-suffix', '--suffix', default='', type=str, help='suffix')

# model configuration
parser.add_argument('-data', '--dataset', default='CIFAR10-DVS', type=str, help='dataset')
parser.add_argument('-arch', '--model', default='vgg11', type=str, help='model')
parser.add_argument('-id', '--identifier', type=str, default="vgg11_L[8]_Training of VGG11 on CIFAR10-DVS",
                    help='model statedict identifier')

# test configuration
parser.add_argument('-dev', '--device', default='0', type=str, help='device')
parser.add_argument('-T', '--time', default=0, type=int, help='snn simulation time')
parser.add_argument('-L', '--L', default=8, type=int, help='Step L')
parser.add_argument("--imChannels", type=int, default=2,
                    help="输入的图像的通道数，rgb 图像为 3， dvs 数据位 2")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    global args
    seed_all(args.seed)
    # preparing data
    # 对于 DVS 数据集，形状为 (T, B, 2, H, W).
    train_loader, test_loader = datapool(DATANAME=args.dataset,
                                         batchsize=args.batch_size,
                                         T=args.time,
                                         seed=args.seed)
    # preparing model
    model = modelpool(args.model, args.dataset,  inChannels=args.imChannels)

    model_dir = os.path.join("./logs",
                             "_D-{args.dataset"
                             "_N-{model}")
    state_dict = torch.load(os.path.join(model_dir, args.identifier + '.pth'), map_location=torch.device('cpu'))

    # if old version state_dict
    keys = list(state_dict.keys())
    for k in keys:
        if "relu.up" in k:
            state_dict[k[:-7] + 'act.thresh'] = state_dict.pop(k)
        elif "up" in k:
            state_dict[k[:-2] + 'thresh'] = state_dict.pop(k)

    model.load_state_dict(state_dict)

    model.to(device)

    model.set_T(args.time)
    model.set_L(args.L)

    # for m in model.modules():
    #     if isinstance(m, IF):
    #         print(m.thresh)

    acc = val(model, test_loader, device, args.time)
    print(acc)


if __name__ == "__main__":
    main()
