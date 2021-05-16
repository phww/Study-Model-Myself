#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/13 下午7:50
# @Author : PH
# @Version：V 0.1
# @File : dataset.py
# @desc :
import os
import numpy as np
from torch.utils.data import Dataset
import PIL.Image as Image


class Airplane(Dataset):
    def __init__(self, root, transforms):
        super(Airplane, self).__init__()
        self.root = root
        self.img_paths = os.listdir(root)
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.img_paths[idx])
        img_name = img_path.split("/")[-1]  # 暂时没用上
        label = 0
        # 按照img路径名的是否含有特定字符，设定标签
        if "-0-" in img_path:
            label = 0
        elif "-1-" in img_path:
            label = 1
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        if self.transforms:
            img = self.transforms(img)
        return img, label


if __name__ == "__main__":
    # 测试
    from torch.utils.data import DataLoader

    dataset = Airplane("./region", None)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True, pin_memory=True)
    for img, label in dataloader:
        print(img.shape)
        print(label)
        break
