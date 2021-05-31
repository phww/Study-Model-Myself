#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/31 下午2:43
# @Author : PH
# @Version：V 0.1
# @File : dataset.py
# @desc : 自己定义的dataset
import random
import nltk
import torch
from torch.utils.data import Dataset, DataLoader
import json
import pickle
import os
from PIL import Image
from generateVocab import Vocabulary


class MyCaptionDataset(Dataset):
    def __init__(self, vocab_path, image_root, caption_path, transforms, max_length=20):
        super(MyCaptionDataset, self).__init__()
        # 读取各种文件
        self.vocab = Vocabulary()
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.img_root = image_root
        self.img_paths = os.listdir(image_root)
        with open(caption_path, "rb") as f:
            self.text_dicts = json.load(f)["sentences"]
        self.transforms = transforms
        self.max_length = max_length

    def __getitem__(self, idx):
        # 按下标索引图片
        img_name = self.img_paths[idx]
        img = Image.open(os.path.join(self.img_root, img_name)).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        # 按下标索引图片的描述，ground truth
        caption = []
        sen_id = random.choice(range(5))
        for i, text_dict in enumerate(self.text_dicts):
            if text_dict["video_id"] in img_name and text_dict["sen_id"] == sen_id:
                caption = text_dict["caption"]
                break
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        target = []
        target.append(self.vocab.word2idx["<start>"])
        target.extend(self.vocab.word2idx[word] for word in tokens)
        target.append(self.vocab.word2idx["<end>"])
        target = torch.tensor(target)
        return img, target

    def __len__(self):
        return len(self.img_paths)


def collect_fn(data):
    """
    因为caption的长短不一，而Dataset要求数据的形状是一样的
    为此需要为Dataset重写一个堆叠函数collect_fn
    Args:
        data: list
            Dataset按下标索引返回的一个batch_size的对象，即长度为batch_size的(img, target)列表
    """
    # 先按target的长度从大到小排序data
    data.sort(key=lambda x: len(x[1]), reverse=True)
    # 图片的堆叠方式不变
    imgs, captions = zip(*data)
    imgs = torch.stack(imgs, dim=0)
    # caption以最长的语句为标准，因为定义了"<pad>"字符的idx为0。在不够长度的在句子后面填0，
    lengths = [len(caption) for caption in captions]
    max_length = max(lengths)
    targets = torch.zeros(len(captions), max_length).long()
    for i, caption in enumerate(captions):
        cur_len = lengths[i]
        targets[i, :cur_len] = caption[:cur_len]
    return imgs, targets, lengths


def get_loader(vocab_path, image_root, caption_path, batch_size, transforms):
    data_set = MyCaptionDataset(vocab_path, image_root, caption_path, transforms=transforms)
    data_loader = DataLoader(data_set, batch_size, shuffle=True, pin_memory=True, collate_fn=collect_fn)
    return data_loader


if __name__ == "__main__":
    # test
    import matplotlib.pyplot as plt
    from torchvision.transforms import ToTensor

    vocab_path = "./video/vocab.pkl"
    image_root = "./generateImgs"
    caption_path = "./video/video_demo/demo.json"
    data_loader = get_loader(vocab_path, image_root, caption_path, batch_size=5, transforms=ToTensor())
    for imgs, targets, lengths in data_loader:
        targets = targets.tolist()
        plt.imshow(imgs[2].permute(1, 2, 0))
        plt.show()
        for word in targets[2]:
            print(data_loader.dataset.vocab.idx2word[word], end=" ")
        break