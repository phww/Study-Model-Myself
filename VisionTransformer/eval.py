#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/16 下午5:16
# @Author : PH
# @Version：V 0.1
# @File : eval.py
# @desc :
import torch
import torchvision.utils
from matplotlib import pyplot as plt
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from model_my import VisionTransformer
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
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
VIT = VisionTransformer(**custom_config)
state = torch.load('./check_point/best.pth')
VIT.load_state_dict(state['model'])
test_set = FashionMNIST(root='/home/ph/Desktop/Tutorials/Pytorch/data',
                        train=False,
                        transform=ToTensor(),
                        download=True)
test_loader = DataLoader(test_set, batch_size=25, shuffle=True)
test_iter = iter(test_loader)


def predict(test_iter, epochs=10, show_img=False):
    avg_correct = 0.
    for epoch in range(epochs):
        inp, labels = next(test_iter)
        pred = VIT(inp).argmax(dim=-1)
        correct = (pred == labels).sum() / len(inp)
        avg_correct += correct
        print(f"epoch:{epoch}   correct:{correct:.2f}")
    print("-" * 30)
    print(f"avg_correct:{avg_correct / epochs:.2f}")
    if show_img:
        inp, labels = next(test_iter)
        pred = VIT(inp).argmax(dim=-1)
        fig = plt.figure(figsize=(20, 20))
        for i in range(inp.size(0)):
            fig.add_subplot(5, inp.size(0) // 5, i + 1)
            if pred[i] != labels[i]:
                plt.xlabel(f"pred:{labels_map[pred[i].item()]} label:{labels_map[labels[i].item()]}", fontsize=15,
                           c='r')
            else:
                plt.xlabel(f"pred:{labels_map[pred[i].item()]} label:{labels_map[labels[i].item()]}", fontsize=15,
                           c='g')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(inp[i].squeeze(), 'gray')
        t = time.strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig("./pred_img/" + t)
        plt.show()


if __name__ == "__main__":
    predict(test_iter, show_img=True)
