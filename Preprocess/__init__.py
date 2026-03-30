from .getdataloader import *


def datapool(DATANAME: str,
             batchsize: int,
             T: int,
             seed: int):
    if DATANAME.lower() == 'cifar10':
        return GetCifar10(batchsize)
    elif DATANAME.lower() == 'cifar100':
        return GetCifar100(batchsize)
    elif DATANAME.lower() == 'imagenet':
        return GetImageNet(batchsize)
    elif DATANAME == 'CIFAR10-DVS':
        return GetCifar10_DVS(batchsize=batchsize,
                              T=T,
                              seed=seed)


    else:
        print("still not support this model")
        exit(0)
