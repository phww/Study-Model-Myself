#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/17 下午2:02
# @Author : PH
# @Version：V 0.1
# @File : dataset.py
# @desc :
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image
import numpy as np
from torchvision.transforms import ToTensor, Compose, Lambda, Resize, ToPILImage
# 导入CitySpace官方的label转换的工具
from labels import labels


class BasicDataset(Dataset):
    def __init__(self, img_root, mask_root, transform=None, target_transform=None):
        """

        Args:
            img_root: 图片root
            mask_root: 标签root
            transform: 输入数据的Transform，一般用ToTensor()加Resize()
            target_transform: 标签的Transform，多分类的标签一定不要用ToTensor()处理，要与输入的Transform区分开
        """
        super(BasicDataset, self).__init__()
        self.img_root = img_root
        self.mask_root = mask_root
        self.transforms = transform
        self.target_transform = target_transform
        # 将图片名保存在列表中
        self.images = os.listdir(img_root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        pass


class CarVanaDataset(BasicDataset):
    """carvana数据集"""

    def __getitem__(self, idx):
        # 组合图片路径
        img_path = os.path.join(self.img_root, self.images[idx])
        mask_path = os.path.join(self.mask_root, self.images[idx].replace(".jpg", "_mask.gif"))
        # 无法直接使用torch.tensor转换，先用np转换
        # opencv 中的imread无法读取Gif格式的图！！！，只能用PIL了
        img = np.array(Image.open(img_path).convert("RGB"))
        # mask一定要转换为“L”模式的灰度图！！
        mask = np.array(Image.open(mask_path).convert("L"))
        # n_classes==1时模型的x，和y都做相同的变换
        if self.transforms is not None:
            img = self.transforms(img)
            mask = self.transforms(mask)
        return img, mask


class CitySpaceDataset(BasicDataset):
    """cityspace数据集"""

    @staticmethod
    def get_file_list(root, keyword):
        """
        获取图片文件的全部路径
        Args:
            root: str
                数据集的根目录
            keyword: str
                用于筛选路径中带keyword的图片路径，不需要时设置为None

        Returns:
            file_list：list
                root目录下全部的图片路径的列表
        """
        file_list = []
        for root, dirs, files in os.walk(root):
            for filename in files:
                if keyword is not None:
                    if keyword in filename:
                        file_list.append(os.path.join(root, filename))
                else:
                    file_list.append(os.path.join(root, filename))
        return file_list

    def __len__(self):
        return len(self.get_file_list(self.img_root, keyword=None))

    def __getitem__(self, idx):
        img_file_list = self.get_file_list(self.img_root, keyword=None)
        mask_file_list = self.get_file_list(self.mask_root, keyword="labelIds")
        img_file_list = sorted(img_file_list)
        mask_file_list = sorted(mask_file_list)
        img_path = img_file_list[idx]
        mask_path = mask_file_list[idx]
        # 根据实际图片的格式转换数据类型
        img = np.array(Image.open(img_path).convert("RGBA"))
        # mask一定要转换为“L”模式的灰度图！！
        mask = np.array(Image.open(mask_path).convert("L"))

        # 原本的mask中的值为【-1，33】的int，每个数字对应该像素对应的类别
        # 但是34类太多了，比如希望将在训练时将car、bus都归为vehicle,为此需要做id到categoryId的映射处理
        ids = {label.id: label for label in labels}  # 使用labels找出所有id以及对应的其他信息的字典

        # 待优化，慢！
        mask_set = set(mask.ravel())  # 统计mask中出现了那些id
        # 创建一个和一样的mask的临时变量
        mask_temp = mask  # 比如先修改id等于26的像素为3，后面如果又修改id等于3的像素为0时就会发生错误
        for id in mask_set:
            mask[mask_temp == id] = ids[id].categoryId
        # n_classes != 1的任务，mask要保持为long，img要归一化，且数据类型为float。因此Transform需要分开做
        if self.transforms is not None:
            img = self.transforms(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
            mask = np.array(mask)
        return img, mask


if __name__ == "__main__":
    target_transforms = Compose([
        ToPILImage(),
        Resize((256, 256))
    ])
    # dataset = CarVanaDataset("/home/ph/Dataset/Carvana/train",
    #                          "/home/ph/Dataset/Carvana/train_mask", transform=ToTensor())
    dataset = CitySpaceDataset("/home/ph/Dataset/CitySpace/leftImg8bit/train",
                               "/home/ph/Dataset/CitySpace/gtFine/train",
                               transform=ToTensor(),
                               target_transform=target_transforms)

    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
    for batch, (img, mask) in enumerate(dataloader):
        print(img.shape)
        print(mask.shape)
        print(mask[0].min(), mask[0].max())
        plt.imshow(mask[2].squeeze().to(torch.long), 'gray')
        plt.show()
        break
