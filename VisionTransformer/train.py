#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/15 下午9:09
# @Author : PH
# @Version：V 0.1
# @File : train.py
# @desc : 使用FashionMNIST训练
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from model_my import VisionTransformer
from utils.template import TemplateModel

# 超参数
custom_config = {
    "img_size": 28,
    "n_classes": 10,
    "img_channels": 1,
    "patch_size": 7,
    "n_dim": 768,
    "n_depth": 12,
    "n_head": 12,
    "use_bias": True,
    "mlp_ratio": 4,
}
epochs = 20

# dataset
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

train_set = datasets.FashionMNIST(root='/home/ph/Desktop/Tutorials/Pytorch/data',
                                  train=True,
                                  transform=ToTensor(),
                                  download=True)
test_set = datasets.FashionMNIST(root='/home/ph/Desktop/Tutorials/Pytorch/data',
                                 train=False,
                                 transform=ToTensor(),
                                 download=True)
train_loader = DataLoader(train_set, batch_size=120, shuffle=True, drop_last=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=120, shuffle=True, drop_last=True, num_workers=4)

# model
VIT = VisionTransformer(**custom_config)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(VIT.parameters(), lr=1e-5)


# trainer
class Trainer(TemplateModel):
    def __init__(self):
        super(Trainer, self).__init__()
        # tensorboard
        self.writer = SummaryWriter()
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_acc = 0.0
        # 模型架构
        self.model = VIT
        self.optimizer = optimizer
        self.criterion = loss_fn
        # 数据集
        self.train_loader = train_loader
        self.test_loader = test_loader
        # 运行设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # check_point 目录
        self.ckpt_dir = "./check_point"
        # 训练时print的间隔
        self.log_per_step = 100


def train(continue_training=False, continue_model=None):
    trainer = Trainer()
    trainer.check_init()
    # trainer.get_model_info(fake_inp=torch.randn(1, 1, 32, 32))
    if continue_training:
        trainer.load_state(continue_model)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        trainer.train_loop()
        trainer.eval(save_per_epochs=5)


if __name__ == "__main__":
    train()
