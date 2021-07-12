#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/20 下午9:47
# @Author : PH
# @Version：V 0.1
# @File : visualize.py
# @desc :
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage
from labels import labels

device = "cuda" if torch.cuda.is_available() else "cpu"
transforms = Compose([
    ToTensor(),
    Resize((256, 256)),
])


def vis(pred_mask, save_path):
    class2color = {label.categoryId: label for label in labels}
    pred_mask = pred_mask.int().cpu().numpy()
    r_mask = pred_mask.copy()
    g_mask = pred_mask.copy()
    b_mask = pred_mask.copy()
    rgb = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
    for idx in range(14):
        r, g, b = class2color[idx].color
        r_mask[pred_mask == idx] = r
        g_mask[pred_mask == idx] = g
        b_mask[pred_mask == idx] = b
        rgb[..., 0] = r_mask
        rgb[..., 1] = g_mask
        rgb[..., 2] = b_mask
    rgb = rgb.astype(np.int32)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(rgb)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    model = torch.load('checkpoint/CitySpaceDataset-cls-142021-04-22+19-53-10.pth')
    inp = np.array(Image.open("./test.png").convert("RGBA"))  # 显示转换为RGBA
    inp = transforms(inp).unsqueeze(dim=0).to(device, dtype=torch.float32)  # 一些预处理
    pred = model(inp)  # 输入网络
    pred = Resize((1024, 2048))(pred)  # Resize回原始尺寸
    pred_mask = torch.log_softmax(pred, dim=1).argmax(dim=1).squeeze()  # 预测标签
    plt.imshow(pred_mask.cpu(), 'gray')
    plt.show()
    vis(pred_mask, "./out.png")
