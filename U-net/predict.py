#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/18 下午4:07
# @Author : PH
# @Version：V 0.1
# @File : predict.py
# @desc :
import numpy as np
import torch
import PIL.Image as Image
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from dataset import CarVanaDataset, CitySpaceDataset
from torch.utils.data import DataLoader
from visualize import vis

dataset = "CitySpaceDataset"
if dataset == "CarVanaDataset":
    img_weight = 1918
    img_height = 1280
else:  # "CitySpaceDataset"
    img_weight = 2048
    img_height = 1024
device = "cuda" if torch.cuda.is_available() else "cpu"
target_transforms = Compose(
    [
        ToPILImage(),
        Resize((img_height, img_weight))
    ]
)
transforms = Compose([
    ToTensor(),
    Resize((256, 256))
])


def predict(model, test_loader, save_path):
    model.eval()
    with torch.no_grad():
        for batch, (img, _) in enumerate(test_loader):
            img = img.to(device, dtype=torch.float32)
            pred = model(img)
            if model.n_classes == 1:
                pred = Resize((1280, 1918))(pred).cpu()
                pred = torch.sigmoid(pred)
                pred = (pred > 0.5).float().cpu()
            else:
                pred = Resize((1024, 2048))(pred).cpu()
                pred = torch.log_softmax(pred, dim=1)
                pred = pred.argmax(dim=1)
            if model.n_classes == 1:
                img_grid = make_grid(pred, nrow=8).mean(dim=0)
                plt.imshow(img_grid.squeeze(), "gray")
                plt.show()
            else:
                plt.imshow(pred[0].squeeze(), "gray")
                vis(pred_mask=pred[0], save_path=None)
                plt.show()
            #     save_path=save_path + str(batch) + "rgb"
            # plt.savefig(save_path + str(batch) + "gray")


# test_set = CarVanaDataset("/home/ph/Dataset/Carvana/test",
#                           "/home/ph/Dataset/Carvana/test_mask", transform=transforms)
test_set = CitySpaceDataset("/home/ph/Dataset/CitySpace/leftImg8bit/val",
                            "/home/ph/Dataset/CitySpace/gtFine/val",
                            transform=transforms,
                            target_transform=target_transforms)
test_loader = DataLoader(test_set, batch_size=16, num_workers=2)
model = torch.load('checkpoint/CitySpaceDatasetcls14.pth').to(device)
predict(model, test_loader, "./pred_imgs/cityspace/img")
