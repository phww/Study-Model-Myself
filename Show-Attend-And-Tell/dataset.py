#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/31 下午2:43
# @Author : PH
# @Version：V 0.1
# @File : dataset.py
# @desc : 自己定义的dataset
import random
import nltk
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import pickle
import os
from tqdm.auto import tqdm
from PIL import Image
from generateVocab import Vocabulary


class MyCaptionDatasetRaw(Dataset):
    def __init__(self, vocab_path, image_root, caption_path, transforms):
        super(MyCaptionDatasetRaw, self).__init__()
        # 读取各种文件
        self.vocab = Vocabulary()
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.img_root = image_root
        self.img_paths = os.listdir(image_root)
        with open(caption_path, "rb") as f:
            self.text_dicts = json.load(f)["sentences"]
        self.transforms = transforms

    def __getitem__(self, idx):
        # 按下标索引图片
        img_name = self.img_paths[idx]
        img = Image.open(os.path.join(self.img_root, img_name)).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        # 按下标索引图片的描述/caption，ground truth
        caption = []
        # 每个图片有5个caption，随机选一个
        sen_id = random.choice(range(5))
        for i, text_dict in enumerate(self.text_dicts):
            # 按图片名暴力搜索对应的caption
            if text_dict["video_id"] in img_name and text_dict["sen_id"] == sen_id:
                caption = text_dict["caption"]
                break
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())

        target = []
        target.append(self.vocab.word2idx["<start>"])
        # 小心append会将caption正文当做一个列表加入到target中！这里用extend更符合要求
        target.extend(self.vocab.word2idx[word] for word in tokens)
        # 当generateVocab.py里面设置词频阈值大于0时。语料库中的一些词不会被记录到字典中。因此设置其为<unk>
        for word in tokens:
            if word not in self.vocab.word2idx.keys():
                word = "<unk>"
            target.append(self.vocab.word2idx[word])
        target.append(self.vocab.word2idx["<end>"])
        target = torch.tensor(target)
        return img, target

    def __len__(self):
        return len(self.img_paths)


class MyCaptionDatasetFusion(Dataset):
    def __init__(self, vocab_path, image_root, caption_path, transforms, k=8):
        super(MyCaptionDatasetFusion, self).__init__()
        self.k = k
        # 读取各种文件
        self.vocab = Vocabulary()
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.img_root = image_root
        self.img_paths = os.listdir(image_root)
        self.img_paths = sorted(self.img_paths)  # 先排序图片路径
        with open(caption_path, "rb") as f:
            self.text_dicts = json.load(f)["sentences"]
        self.transforms = transforms

    def __getitem__(self, idx):
        # 按下标索引视频提取的所有帧，因为提前排过序。[idx*k，（idx+1）*k]的图片都属于一个视频中提取的关键帧
        video_id = self.img_paths[idx * self.k].split("-")[0]  # G_16000-1.jpg -> G_16000
        img = Image.open(os.path.join(self.img_root, self.img_paths[idx * self.k])).convert('RGB')
        if self.transforms is not None:
            imgs = self.transforms(img)  # 3, 224, 224
        for i in range(1, self.k):
            img_name = self.img_paths[idx * self.k + i]
            img = Image.open(os.path.join(self.img_root, img_name)).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
                imgs = torch.cat([imgs, img], dim=0)  # 3*k, 224, 224

        # 按下标索引图片的描述/caption，ground truth
        caption = []
        # 训练时。每个视频有5个caption，随机选一个
        sen_id = random.choice(range(5))
        for i, text_dict in enumerate(self.text_dicts):
            # 按图片名暴力搜索对应的caption。（这里可以二分搜索优化一下）
            if text_dict["video_id"] == video_id and text_dict["sen_id"] == sen_id:
                caption = text_dict["caption"]

        # tokenize 训练时随机选取的caption
        target = self.tokenize_caption(caption)
        return imgs, target, video_id

    def tokenize_caption(self, caption):
        target = []
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        target.append(self.vocab.word2idx["<start>"])
        # 当generateVocab.py里面设置词频阈值大于0时。语料库中的一些词不会被记录到字典中。因此设置其为<unk>
        for word in tokens:
            if word not in self.vocab.word2idx.keys():
                word = "<unk>"
            target.append(self.vocab.word2idx[word])
        target.append(self.vocab.word2idx["<end>"])
        target = torch.tensor(target)
        return target

    def __len__(self):
        return len(self.img_paths) // self.k


def collect_fn(data):
    """
    因为caption的长短不一，而Dataset要求数据的形状是一样的
    为此需要为Dataset重写一个堆叠函数collect_fn
    Args:
        data: list
            Dataset按下标索引返回的一个batch_size的对象，即长度为batch_size的(img, target)列表
    Returns:
        按照一个批次的caption的长度排序后的数据：
        imgs: shape"B 3*k 224 224"
        targets: 经过Vocab编码和填充过后的caption shape"B max_length"
        lengths: caption的原始长度。包含<start>和<end>
    """
    # 先按target的长度从大到小排序data
    data.sort(key=lambda x: len(x[1]), reverse=True)

    # 图片和视频名的堆叠方式不变
    all_video_id = []
    imgs, captions, video_ids = zip(*data)
    imgs = torch.stack(imgs, dim=0)


    # caption以最长的语句为标准，因为定义了"<pad>"字符的idx为0。在不够长度的在句子后面填0，
    lengths = [len(caption) for caption in captions]

    # 用Pytorch提供的API填充语句
    targets = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
    # 自己写也很容易
    # max_length = max(lengths)
    # targets = torch.zeros(len(captions), max_length).long()
    # for i, caption in enumerate(captions):
    #     cur_len = lengths[i]
    #     targets[i, :cur_len] = caption[:cur_len]

    return imgs, targets, lengths, video_ids


def get_loader(vocab_path, image_root, caption_path, batch_size, transforms, use_fusion=True):
    """
    返回数据集的dataloader
    Args:
        vocab_path: 预处理得到的字典文件path.格式为pkl
        image_root: 预处理视频得到的关键帧图片的root
        caption_path: 原始的caption文件。格式为json

    Returns:
        dataloader:dataset的迭代器，每次迭代返回一个批次的imgs, targets, lengths
    """
    if use_fusion:
        data_set = MyCaptionDatasetFusion(vocab_path, image_root, caption_path, transforms=transforms)
    else:
        data_set = MyCaptionDatasetRaw(vocab_path, image_root, caption_path, transforms=transforms)
    data_loader = DataLoader(data_set, batch_size, shuffle=True, pin_memory=True, collate_fn=collect_fn, num_workers=6)
    return data_loader


if __name__ == "__main__":
    # test
    import matplotlib.pyplot as plt
    from torchvision.transforms import ToTensor, Resize, Compose

    transforms = Compose([Resize((224, 224)),
                          ToTensor()])
    vocab_path = "/home/ph/Dataset/VideoCaption/vocab.pkl"
    image_root = "/home/ph/Dataset/VideoCaption/generateImgs/train"
    caption_path = "/home/ph/Dataset/VideoCaption/info.json"
    data_loader = get_loader(vocab_path, image_root, caption_path, batch_size=10, transforms=transforms,
                             use_fusion=True)
    cnt = 0
    mean = np.zeros((1, 3))
    std = np.zeros((1, 3))
    for imgs, targets, lengths, video_ids in data_loader:
        cnt += 1
        imgs = imgs.numpy()
        print(imgs.shape)
        # mean += imgs.mean(axis=(0, 2, 3))
        # std += imgs.std(axis=(0, 2, 3))
        # mean /= cnt
        # std /= cnt
        # print(mean)  # 0.43710339 0.41183448 0.39289876
        # print(std)  # 0.27540463 0.27135348 0.27471914

        batch_captions = []
        for video_id in video_ids:
            captions = []
            for test_dict in data_loader.dataset.text_dicts:
                if test_dict["video_id"] == video_id:
                    captions.append(test_dict['caption'])
            batch_captions.append(captions)
        print(batch_captions[2])
        targets = targets.tolist()
        print(video_ids[2])
        plt.imshow(imgs[2].transpose(1, 2, 0).mean(axis=-1), "gray")
        for word in targets[2]:
            print(data_loader.dataset.vocab.idx2word[word], end=" ")
        plt.show()
        break
