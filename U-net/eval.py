#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/18 下午3:47
# @Author : PH
# @Version：V 0.1
# @File : eval.py
# @desc :
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def f(pred, target):
    i = (pred == target).sum()
    u = pred.numel() + target.numel()
    dice_loss = 2 * i / u
    TP = ((pred == 1) & (target == 1)).sum()
    FP = ((pred == 1) & (target == 0)).sum()
    FN = ((pred == 0) & (target == 1)).sum()
    TN = ((pred == 0) & (target == 0)).sum()
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    return P, R, F1, dice_loss


def diceLoss(pred, target, ep=1e-7):
    i = (pred * target).sum(dim=(1, 2, 3))
    u = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    loss = ((2 * i + ep) / (u + ep)).mean()
    return 1 - loss


def diceLoss_v2(pred, target, ep=1e-7):
    batch_size = pred.size(0)
    pred = pred.contiguous().view(batch_size, -1)
    target = target.contiguous().view(batch_size, -1)
    i = torch.mul(pred, target).sum(dim=1)
    u = pred.sum(dim=0) + target.sum(dim=1)
    loss = ((2 * i + ep) / (u + ep)).mean()
    return 1 - loss


def miou():
    pass


def eval(model, dataloader):
    model.eval()
    num_correct = 0.
    num_pixels = 0.
    dice_loss = 0.
    dice_loss2 = 0.
    R = 0.
    P = 0.
    F1 = 0.
    dice_loss_v3 = 0.
    mask_type = torch.float32 if model.n_classes == 1 else torch.long
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=mask_type)
            pred = model(x)
            if model.n_classes == 1:
                pred = torch.sigmoid(pred)
                pred = (pred > 0.5).float()
            else:
                pred = torch.softmax(pred, dim=1)
                pred = pred.argmax(dim=1)
            # 计算准确率
            num_correct += (pred == y).sum()
            num_pixels += torch.numel(pred)
            # 计算dice_loss
            # dice_loss += diceLoss(pred, y)
    #         dice_loss2 += diceLoss_v2(pred, y)
    #         R, P, F1, dice_loss_v3 = R + f(pred, y)[0], P + f(pred, y)[1],F1 + f(pred, y)[2], dice_loss_v3 + f(pred, y)[3]
    acc = num_correct / num_pixels
    size = len(dataloader.dataset)
    # dice_loss /= size
    # dice_loss2 /= size
    # R, P, F1, dice_loss_v3 = R/size, P/size, F1/size, dice_loss_v3/size
    return acc, dice_loss  # , dice_loss2, R, P, F1, dice_loss_v3


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage
    from dataset import CarVanaDataset

    transforms = Compose([
        ToTensor(),
        Resize((256, 256))
    ])
    target_transforms = Compose([
        ToPILImage(),
        Resize((256, 256))
    ])

    model = torch.load('checkpoint/CarVanaDatasetcls12021-04-22+17-04-43.pth')
    dataset = CarVanaDataset("/home/ph/Dataset/Carvana/test",
                             "/home/ph/Dataset/Carvana/test_mask",
                             transform=transforms,
                             target_transform=target_transforms)
    dataloader = DataLoader(dataset)
    acc, loss, loss2, R, P, F1, loss3 = eval(model, dataloader)
    print(acc, loss, loss2, R, P, F1, loss3)
