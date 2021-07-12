#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/17 下午8:23
# @Author : PH
# @Version：V 0.1
# @File : train.py
# @desc :
import os.path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage
from custom_model import Unet
from dataset import CarVanaDataset, CitySpaceDataset
from eval import eval
import time
import torch.utils.tensorboard as tb
import logging

logging.basicConfig(level="INFO")

# 每次训练时根据实际情况修改的参数
dataset = "CitySpaceDataset"
logging.info(f"Training model use {dataset}")
checkpoint = None  # 使用之前的模型继续训练,不使用：None，使用模型路径的str
writer = tb.SummaryWriter(log_dir="./runs/City")

# 数据集path
if dataset == "CarVanaDataset":
    train_path = "/home/ph/Dataset/Carvana/train"
    test_path = "/home/ph/Dataset/Carvana/test"
    train_mask_path = "/home/ph/Dataset/Carvana/train_mask"
    test_mask_path = "/home/ph/Dataset/Carvana/test_mask"
    img_weight = 1918
    img_height = 1280
elif dataset == "CitySpaceDataset":
    train_path = "/home/ph/Dataset/CitySpace/leftImg8bit/train"
    test_path = "/home/ph/Dataset/CitySpace/leftImg8bit/val"
    train_mask_path = "/home/ph/Dataset/CitySpace/gtFine/train"
    test_mask_path = "/home/ph/Dataset/CitySpace/gtFine/val"
    img_weight = 2048
    img_height = 1024

# 超参数
n_classes = 14  # 每次训练根据实际情况修改
in_channels = 4  # 每次训练根据实际情况修改
epochs = 6
batch_size = 20
learning_rate = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
transforms = Compose([
    # ToTensor相当于将HWC转换为CHW，并将0-255的像素值归一化到0-1
    ToTensor(),
    # resize会使模型少一些精度，但是可以加快训练速度，减小显存压力
    Resize((256, 256)),
])
target_transforms = Compose([
    # resize只能处理PIL.Image或tensor类型的数据。而在dataset中的输出为nd.array类型的数据
    # 因此要先用ToPILImage将nd.array转换为PIL格式
    # 千万不要对mask使用ToTensor，该方法会归一化像素，千万不要归一化mask！！！
    ToPILImage(),
    Resize((256, 256))
])


def train_loop(model, train_loader, loss_fn, optimizer, print_pre_batch=10):
    model.train()
    running_loss = 0.0
    # 二分类和多分类的损失函数不一样，对输入的要求也不一样
    # 多分类使用交叉熵损失标签要求数据类型为long
    # 二分类使用BCEWithLogitsLoss损失，数据类型为float
    mask_type = torch.float32 if model.n_classes == 1 else torch.long
    for batch, (x, y) in enumerate(train_loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=mask_type)
        # forward
        pred = model(x)
        loss = loss_fn(pred, y)
        writer.add_scalar("loss/train", loss)
        running_loss += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # visualize
        if batch % print_pre_batch == 0 and batch != 0:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy())
                writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy())
            print(f"loss:{running_loss / batch * len(x):.2f}  "
                  f"batch:[{batch * len(x):>5d}/{len(train_loader) * len(x):>5d}]")
        running_loss = 0.0


# 训练
torch.cuda.empty_cache()

# 模型
if checkpoint:
    model = torch.load(checkpoint)
else:
    model = Unet(in_channels, n_classes).to(device)
logging.info(f"Load model use {checkpoint}")

# 数据集
if dataset == "CarVanaDataset":
    train_set = CarVanaDataset(train_path, train_mask_path, transform=transforms)
    test_set = CarVanaDataset(test_path, test_mask_path, transform=transforms)
elif dataset == "CitySpaceDataset":
    train_set = CitySpaceDataset(train_path, train_mask_path, transform=transforms, target_transform=target_transforms)
    test_set = CitySpaceDataset(test_path, test_mask_path, transform=transforms, target_transform=target_transforms)
train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size, shuffle=True, num_workers=4)
logging.info(f"Load dataset use {dataset}")

# 损失函数
# 二分类使用sigmoid函数处理后，使用逻辑回归的损失函数
if n_classes == 1:
    loss_fn = nn.BCEWithLogitsLoss()
# 多分类使用softmax函数处理后，使用交叉熵损失函数
else:
    loss_fn = nn.CrossEntropyLoss()
logging.info(f"Loss use {loss_fn}")

# 优化器
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
logging.info(f"Optimizer use {optimizer}")

# 训练循环
for epoch in range(epochs):
    print("epoch:", epoch)
    train_loop(model, train_loader, loss_fn, optimizer)
    acc, dice_loss = eval(model, test_loader)
    writer.add_scalar("eval_acc", acc, global_step=epoch)
    # writer.add_scalar("eval_diceLoss", dice_loss.item(), global_step=epoch)
    print(f"eval_ACC:{acc:.5f}    diceLoss:{dice_loss:.5f}")
    print("*" * 30)
logging.info(f"Training Finished!")

# 保存模型
t = time.strftime("%Y-%m-%d+%H-%M-%S")
save_path = os.path.join("checkpoint", dataset + "-cls-" + str(n_classes) + t + ".pth")
torch.save(model, save_path)
logging.info(f"Save model in {save_path}")
