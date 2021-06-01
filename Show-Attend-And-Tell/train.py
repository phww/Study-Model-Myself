#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/31 下午3:53
# @Author : PH
# @Version：V 0.1
# @File : train.py
# @desc :
import matplotlib.pyplot as plt
import nltk.translate.bleu_score
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, Resize
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence
from dataset import get_loader
from model import ShowAttendTell, CNNEncoder, RNNDecoderWithAttention
from utils.template import TemplateModel
from generateVocab import Vocabulary

global model, optimizer, loss_fn, train_loader, test_loader


def translate2Sentence(words_vec, vocab, reference):
    sentences = []
    for word_vec in words_vec:
        sentence = []
        for word_idx in word_vec:
            word = vocab.idx2word[word_idx]
            # reference去掉"<start>"和"<pad>"token
            if word not in ["<start>", "<pad>"] and reference:
                sentence.append(word)
            # hypothesis去掉"<pad>"token
            elif word not in ["<pad>"] and not reference:
                sentence.append(word)
        sentences.append(sentence)
    return sentences


class Trainer(TemplateModel):
    def __init__(self, ):
        super(Trainer, self).__init__()
        # tensorboard
        self.writer = SummaryWriter()
        # 模型架构
        self.model_list = [encoder, decoder]
        self.optimizer_list = [encoder_optimizer, decoder_optimizer]
        self.criterion = loss_fn
        # 数据集
        self.train_loader = train_loader
        self.test_loader = test_loader
        # 训练时print的间隔
        self.log_per_step = 5

    # 重载模板中的train_loss_per_batch和metric方法
    def train_loss_per_batch(self, batch):
        # 拆包data_loader返回的对象
        imgs, targets, lengths = batch
        imgs = imgs.to(self.device, dtype=torch.float)
        targets = targets.to(self.device, dtype=torch.long)

        imgs = self.model_list[0](imgs)

        pred_word_vec, caption_embed, decode_length, att_weights = self.model_list[1](imgs, targets, lengths)
        pred_word_vec = pack_padded_sequence(pred_word_vec, decode_length, batch_first=True)[0].to(self.device)
        targets = pack_padded_sequence(targets[:, 1:], decode_length, batch_first=True)[0].to(self.device)
        loss = self.criterion(pred_word_vec, targets)
        loss += 1.0 * ((1. - att_weights.sum(dim=1)) ** 2).mean()
        return loss

    def eval_scores_per_batch(self, batch):
        imgs, targets, lengths = batch
        imgs = imgs.to(self.device)
        targets = targets.to(self.device, dtype=torch.long)
        imgs = encoder(imgs)
        pred_words_vec, caption_embed, decode_length, att_weights = decoder(imgs, targets, lengths)
        pred_words_vec = pred_words_vec.argmax(dim=2)
        hypothesis = translate2Sentence(pred_words_vec.tolist(), vocab=self.train_loader.dataset.vocab, reference=False)
        reference = translate2Sentence(caption_embed.tolist(), vocab=self.train_loader.dataset.vocab, reference=True)
        for i in range(1):
            print(reference[i])
            print(hypothesis[i])
            print("~" * 50)
        scores = self.metric(hypothesis, reference)
        return scores

    def metric(self, hypothesis, reference):
        self.key_metric = "bleu"
        scores = {}
        bleu = 0.0
        # bleu = nltk.translate.bleu_score.corpus_bleu([reference], hypothesis, smoothing_function=None, )
        for idx in range(len(reference)):
            bleu += nltk.translate.bleu_score.sentence_bleu([reference[idx]], hypothesis[idx])
        scores[self.key_metric] = bleu / len(reference)
        return scores


def main(continue_train=False):
    global model, optimizer, loss_fn, train_loader, test_loader, encoder_optimizer, decoder_optimizer, encoder, decoder
    vocab_path = "./video/vocab.pkl"
    image_root = "./generateImgs"
    caption_path = "./video/video_demo/demo.json"
    BATCH_SIZE = 23
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
    if continue_train:
        trainer.load_state("./check_point/epoch7.pth")

    for epoch in range(75):
        trainer.train_loop()
        trainer.eval(save_per_epochs=10)


if __name__ == '__main__':
    main(continue_train=False)
