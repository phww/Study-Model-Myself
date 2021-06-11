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
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, Resize, Normalize
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence
from dataset import get_loader
from model import ShowAttendTell, CNNEncoder, RNNDecoderWithAttention
from utils.template import TemplateModel
from generateVocab import Vocabulary
import pickle

global vocab


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
    def __init__(self, loss_fn, train_loader, test_loader,
                 encoder_optimizer, decoder_optimizer, encoder, decoder):
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
        self.ckpt_dir = "./check_point5"
        #
        self.lr_scheduler_type = "loss"  # None "metric" "loss"

    # 重载模板中的train_loss_per_batch和metric方法
    def loss_per_batch(self, batch):
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
        imgs = self.model_list[0](imgs)
        pred_words_vec, caption_embed, decode_length, att_weights = self.model_list[1](imgs, targets, lengths)
        pred_words_vec = pred_words_vec.argmax(dim=2)
        hypothesis = translate2Sentence(pred_words_vec.tolist(), vocab=vocab, reference=False)
        reference = translate2Sentence(caption_embed.tolist(), vocab=vocab, reference=True)
        self.writer.add_text("hypothesis", text_string=str(hypothesis[0]), global_step=self.global_step_eval)
        self.writer.add_text("reference", text_string=str(reference[0]), global_step=self.global_step_eval)
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


def main(continue_model=None):
    global vocab
    vocab_path = "/home/ph/Dataset/VideoCaption/vocab.pkl"
    vocab = Vocabulary()
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    print("vocab_size:", len(vocab))
    image_root_train = "/home/ph/Dataset/VideoCaption/generateImgs/train"
    image_root_val = "/home/ph/Dataset/VideoCaption/generateImgs/val"
    caption_path = "/home/ph/Dataset/VideoCaption/info.json"
    epochs = 100
    batch_size_train = 200
    batch_size_val = 50
    transforms = Compose([Resize((224, 224)),
                          RandomHorizontalFlip(),
                          ToTensor(),
                          Normalize([0.43710339, 0.41183448, 0.39289876],
                                    [0.27540463, 0.27135348, 0.27471914])])
    encoder_init_lr = 1e-4
    decoder_init_lr = 1e-4
    train_loader = get_loader(vocab_path, image_root_train,
                              caption_path, batch_size=batch_size_train,
                              transforms=transforms)
    val_loader = get_loader(vocab_path, image_root_val,
                            caption_path, batch_size=batch_size_val,
                            transforms=transforms)
    model_config = {"att_dim": 512,
                    "decoder_dim": 512,
                    "embed_dim": 512,
                    "vocab_size": len(vocab)}
    encoder = CNNEncoder()
    decoder = RNNDecoderWithAttention(**model_config)
    encoder_optimizer = torch.optim.Adam(params=encoder.parameters(), lr=encoder_init_lr)
    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=decoder_init_lr)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(loss_fn, train_loader, val_loader,
                      encoder_optimizer, decoder_optimizer,
                      encoder, decoder)
    trainer.check_init(clean_log=True)
    if continue_model is not None:
        trainer.load_state(continue_model)

    for epoch in range(epochs):
        trainer.train_loop()
        trainer.eval(save_per_epochs=10)


if __name__ == '__main__':
    main()
