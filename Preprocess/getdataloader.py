import os
import random
from typing import Tuple
from typing import Union, Dict

import numpy as np
import torch
from PIL.Image import Image
from spikingjelly.datasets import cifar10_dvs
from torch import nn, Tensor
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from Preprocess.augment import Cutout, CIFAR10Policy

# your own data dir
DIR = {'CIFAR10': './datasets', 'CIFAR100': './datasets', 'ImageNet': 'YOUR_IMAGENET_DIR'}


def GetCifar10(batchsize, attack=False):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  Cutout(n_holes=1, length=16)
                                  ])
    if attack:
        trans = transforms.Compose([transforms.ToTensor()])
    else:
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans, download=True)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader


def GetCifar100(batchsize):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                                       std=[n / 255. for n in [68.2, 65.4, 70.4]]),
                                  Cutout(n_holes=1, length=16)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                                     std=[n / 255. for n in [68.2, 65.4, 70.4]])])
    train_data = datasets.CIFAR100(DIR['CIFAR100'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR100(DIR['CIFAR100'], train=False, transform=trans, download=True)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
    return train_dataloader, test_dataloader


def GetImageNet(batchsize):
    trans_t = transforms.Compose([transforms.RandomResizedCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                  ])

    trans = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    train_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'train'), transform=trans_t)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=False, num_workers=8, sampler=train_sampler,
                                  pin_memory=True)

    test_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'val'), transform=trans)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=2, sampler=test_sampler)
    return train_dataloader, test_dataloader


def get_TV_transforms(use_v2: bool):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2

        return torchvision.transforms.v2
    else:
        import torchvision.transforms

        return torchvision.transforms


class DataAugment:
    def __init__(self,
                 nameDataset: str,
                 dictTransforms: Dict,
                 is_train=False,
                 use_v2=True,
                 scale=True,
                 ) -> None:
        r"""
        :param nameDataset:
        :param dictTransforms:
        :param is_train:
        :param use_v2:
        :param scale: whether to scale pixel values (like ImageNet/CIFAR) into [0, 1]
        :param cOut:
        :param numHoles:
        :param length:
        """
        self.cfg = dictTransforms
        self.is_train = is_train
        self.TVTransforms = get_TV_transforms(use_v2)
        self.useTSFv2 = use_v2
        self.scale = scale
        self.nameDataset = nameDataset
        self._init_transforms()
        self._set_transform_strategy()

    def _init_transforms(self):
        self.resize = self.TVTransforms.Resize(
            size=(self.cfg["resize_H"],
                  self.cfg["resize_W"]),
            antialias=True
        )
        self.rotate = self.TVTransforms.RandomRotation(degrees=30)
        self.shearx = self.TVTransforms.RandomAffine(degrees=0, shear=(-30, 30))
        self.flipH = self.TVTransforms.RandomHorizontalFlip()
        self.crop = self.TVTransforms.RandomCrop(self.cfg["resize_H"],
                                                 4)

        # 这里的语句和torchvision版本有关系。0.20以上就不用执行if判断。比如云服务HY2上。
        # 而torchvision 0.15一下的版本就得用if判断。
        # 主要是高版本的torchvision中没有ToImageTensor.
        if self.useTSFv2:
            self.T2T = self.TVTransforms.Compose([self.TVTransforms.ToImage(),
                                                  self.TVTransforms.ToDtype(torch.float32, scale=self.scale)])
        else:
            self.T2T = self.TVTransforms.ToTensor()

    def _set_transform_strategy(self):
        if self.cfg["name"] == "kornia":
            self.transform = nn.Identity()
        else:
            transform_strategies = {
                "sew": self.sew_augment,
                "flip": self.default_augment,
                "nda": self.nda_augment,
                "static": self.static_augment
            }
            self.transform = transform_strategies.get(
                self.cfg["name"], self.default_augment
            )

    @staticmethod
    def roll_fun(data):
        offsets = [random.randint(-5, 5) for _ in range(2)]
        return torch.roll(data, shifts=tuple(offsets), dims=(-2, -1))

    @property
    def default_augment(self):
        transforms = [self.T2T, self.resize]
        if self.is_train:
            # noinspection PyTypeChecker
            transforms.extend([self.flipH, self.roll_fun])
        return self.TVTransforms.Compose(transforms)

    @property
    def static_augment(self):
        transforms = [self.T2T]
        if self.is_train:
            # noinspection PyTypeChecker
            transforms.extend([self.crop, self.flipH])
            if self.cfg["Cout"]:
                if self.nameDataset == "CIFAR100":
                    length = 8
                elif self.nameDataset == "CIFAR10":
                    length = 16
                else:
                    raise NotImplementedError(f"目前只针对 CIFAR-10 和 CIFAR-100 数据集设计了Cout变换")
                # noinspection PyTypeChecker
                transforms.append(Cutout(n_holes=self.cfg["numHoles"],
                                         length=length
                                         ))

            if self.nameDataset == "cifar10":
                # noinspection PyTypeChecker
                transforms.append(self.TVTransforms.Normalize((0.4914, 0.4822, 0.4465),
                                                              (0.2023, 0.1994, 0.2010)), )
            else:
                # noinspection PyTypeChecker
                transforms.append(self.TVTransforms.Normalize((0.5071, 0.4867, 0.4408),
                                                              (0.2675, 0.2565, 0.2761)), )

        else:
            if self.nameDataset == "cifar10":
                # noinspection PyTypeChecker
                transforms.append(self.TVTransforms.Normalize((0.4914, 0.4822, 0.4465),
                                                              (0.2023, 0.1994, 0.2010)), )
            else:
                # noinspection PyTypeChecker
                transforms.append(self.TVTransforms.Normalize((0.5071, 0.4867, 0.4408),
                                                              (0.2675, 0.2565, 0.2761)), )

        return self.TVTransforms.Compose(transforms)

    @property
    def sew_augment(self):
        if not self.is_train:
            return self.TVTransforms.Compose([self.T2T,
                                              self.TVTransforms.Lambda(lambda x: x)])

        def augment(img):
            sec_list = np.random.choice(
                img.shape[0], self.cfg["T_train"], replace=False
            )
            return img[np.sort(sec_list), :]

        return self.TVTransforms.Compose([self.T2T, augment])

    @property
    def nda_augment(self):
        if not self.is_train:
            return self.TVTransforms.Compose([self.T2T,
                                              self.TVTransforms.Lambda(lambda x: x)])

        aug_choice = np.random.choice(["roll", "rotate", "shear"])

        if aug_choice == "roll":
            def roll_fun(data):
                offsets = [random.randint(-5, 5) for _ in range(2)]
                return torch.roll(data, shifts=tuple(offsets), dims=(2, 3))

            return roll_fun
        elif aug_choice == "rotate":
            return self.TVTransforms.Compose([self.rotate])
        else:  # shear
            return self.TVTransforms.Compose([self.shearx])

    def __call__(self, img: Union[np.ndarray, Tensor, Image]) -> Tensor:
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        # print(f"shape before : {img.shape}")
        if self.transform is not nn.Identity():
            # for tfm in self.transform.transforms:
            #     for tf in tfm.transforms:
            #         print(f"{tfm}")
            #         img = tf(img)
            #         print(f"type:{type(img)}")
            img = self.transform(img)
            # print(f"shape:{img.shape}")

        return img


def GetCifar10_DVS(batchsize: int,
                   T: int,
                   seed: int) -> Tuple[DataLoader, DataLoader]:
    augmentTrain = DataAugment(nameDataset="CIFAR10-DVS",
                               dictTransforms={
                                   "name": "Flip_Rotation",
                                   "T_train": None,
                                   "resize_H": 48,
                                   "resize_W": 48,
                                   "Cout": False
                               },
                               is_train=True,
                               scale=False)
    augmentVal = DataAugment(nameDataset="CIFAR10-DVS",
                             dictTransforms={
                                 "name": "Flip_Rotation",
                                 "T_train": None,
                                 "resize_H": 48,
                                 "resize_W": 48,
                                 "Cout": False
                             },
                             is_train=False,
                             scale=False)
    datasetTotal = cifar10_dvs.CIFAR10DVS(root="/home/baiwangyuanfan/Datasets/cifar10-dvs",
                                          data_type="frame",
                                          frames_number=T,
                                          split_by="number")
    train_data, test_data = random_split(
        datasetTotal, [0.9, 0.1], torch.Generator().manual_seed(seed)
    )

    train_data.dataset.transform = augmentTrain
    test_data.dataset.transform = augmentVal
    train_dataloader = DataLoader(train_data,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  num_workers=8,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_data,
                                 batch_size=batchsize,
                                 shuffle=False,
                                 num_workers=4)

    return train_dataloader, test_dataloader
