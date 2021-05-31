#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/31 下午3:53
# @Author : PH
# @Version：V 0.1
# @File : train.py
# @desc :
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, Resize
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence
from dataset import get_loader
from model import ShowAttendTell, CNNEncoder, RNNDecoderWithAttention
from utils.template import TemplateModel

from generateVocab import Vocabulary

global model, optimizer, loss_fn, train_loader, test_loader


class Trainer(TemplateModel):
    def __init__(self, ):
        super(Trainer, self).__init__()
        # tensorboard
        self.writer = SummaryWriter()
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_acc = 0.0
        # 模型架构
        self.model = [encoder, decoder]
        self.optimizer = [encoder_optimizer, decoder_optimizer]
        self.criterion = loss_fn
        # 数据集
        self.train_loader = train_loader
        self.test_loader = test_loader
        # 运行设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # check_point 目录
        self.ckpt_dir = "./check_point"
        # 训练时print的间隔
        self.log_per_step = 5

    # 重载模板中的train_loss_per_batch和metric方法
    def train_loss_per_batch(self, batch):
        # 拆包data_loader返回的对象
        imgs, targets, lengths = batch
        imgs = imgs.to(self.device, dtype=torch.float)
        targets = targets.to(self.device, dtype=torch.long)

        imgs = self.model[0](imgs)

        pred_word_vec, caption_embed, decode_length, att_weights = self.model[1](imgs, targets, lengths)
        pred_word_vec = pack_padded_sequence(pred_word_vec, decode_length, batch_first=True)[0].to(self.device)
        targets = pack_padded_sequence(targets[:, 1:], decode_length, batch_first=True)[0].to(self.device)
        loss = self.criterion(pred_word_vec, targets)
        loss += 1.0 * ((1. - att_weights.sum(dim=1)) ** 2).mean()
        return loss

    # def metric(self, pred, y):

def main():
    global model, optimizer, loss_fn, train_loader, test_loader, encoder_optimizer, decoder_optimizer, encoder, decoder
    vocab_path = "./video/vocab.pkl"
    image_root = "./generateImgs"
    caption_path = "./video/video_demo/demo.json"
    BATCH_SIZE = 5
    TRANSFORMS = Compose([Resize((224, 224)),
                          RandomHorizontalFlip(),
                          ToTensor()])
    LR = 1e-4
    train_loader = test_loader = get_loader(vocab_path, image_root,
                                            caption_path, batch_size=BATCH_SIZE,
                                            transforms=TRANSFORMS)
    model_config = {"att_dim": 512,
                    "decoder_dim": 512,
                    "embed_dim": 512,
                    "vocab_size": len(train_loader.dataset.vocab)}
    encoder = CNNEncoder()
    decoder = RNNDecoderWithAttention(**model_config)
    encoder_optimizer = torch.optim.Adam(params=encoder.parameters(), lr=1e-4)
    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=4e-4)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer()
    trainer.check_init()
    for epoch in range(1):
        trainer.train_loop()

if __name__ == '__main__':
    main()

