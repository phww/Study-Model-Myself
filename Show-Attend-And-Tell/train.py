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
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, \
    RandomVerticalFlip, RandomRotation, Resize, Normalize
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence
from dataset import get_loader
from model import ShowAttendTell, CNNEncoder, RNNDecoderWithAttention
from utils.template import TemplateModel
from generateVocab import Vocabulary
import pickle
import os

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
        self.writer = SummaryWriter(comment=self.ckpt_dir)
        # 模型架构
        self.model_list = [encoder, decoder]
        self.optimizer_list = [encoder_optimizer, decoder_optimizer]
        self.criterion = loss_fn
        # 数据集
        self.train_loader = train_loader
        self.test_loader = test_loader
        # 训练时print的间隔
        self.log_per_step = 20
        self.ckpt_dir = "./check_point_fusion_captions"
        #
        self.lr_scheduler_type = "metric"  # None "metric" "loss"

    # 重载模板中的train_loss_per_batch和metric方法
    def loss_per_batch(self, batch):
        # 拆包data_loader返回的对象
        imgs, targets, lengths, _ = batch
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
        imgs, targets, lengths, video_ids = batch
        imgs = imgs.to(self.device)
        targets = targets.to(self.device, dtype=torch.long)
        imgs = self.model_list[0](imgs)
        pred_words_vec, caption_embed, decode_length, att_weights = self.model_list[1](imgs, targets, lengths)
        pred_words_vec = pred_words_vec.argmax(dim=2)
        hypothesis = translate2Sentence(pred_words_vec.tolist(), vocab=vocab, reference=False)

        batch_captions = []
        for video_id in video_ids:
            captions = []
            for test_dict in self.train_loader.dataset.text_dicts:
                if test_dict["video_id"] == video_id:
                    captions.append(test_dict['caption'])
            batch_captions.append(captions)
        batch_references = []
        for captions in batch_captions:
            references = []
            for caption in captions:
                reference = caption.split(" ")
                reference.append("<end>")
                references.append(reference)
            batch_references.append(references)
        # self.writer.add_text("hypothesis", text_string=str(hypothesis[0]), global_step=self.global_step_eval)
        # self.writer.add_text("reference", text_string=str(reference[0]), global_step=self.global_step_eval)
        scores = self.metric(hypothesis, batch_references)
        return scores

    def metric(self, hypothesis, reference):
        self.key_metric = "bleu"
        scores = {}
        bleu = 0.0
        # bleu = nltk.translate.bleu_score.corpus_bleu([reference], hypothesis, smoothing_function=None, )
        for idx in range(len(reference)):
            bleu += nltk.translate.bleu_score.sentence_bleu(reference[idx], hypothesis[idx])
        scores[self.key_metric] = bleu / len(reference)
        return scores


def getArg():
    import argparse
    parser = argparse.ArgumentParser(description="train model")
    # path
    parser.add_argument("--vocab_path",
                        default="/home/ph/Dataset/VideoCaption/vocab.pkl",
                        help="字典文件的路径")
    parser.add_argument("--image_root_train",
                        default="/home/ph/Dataset/VideoCaption/generateImgs/train",
                        help="训练集图片文件夹的根目录")
    parser.add_argument("--image_root_val",
                        default="/home/ph/Dataset/VideoCaption/generateImgs/val",
                        help="验证集图片文件夹的根目录")
    parser.add_argument("--caption_path",
                        default="/home/ph/Dataset/VideoCaption/info.json",
                        help="视频对应的描述性语句")
    # train config
    parser.add_argument("--epochs", type=int, help="训练的轮次")
    parser.add_argument("--batch_size_train", type=int, default=5, help="训练集的batch_size")
    parser.add_argument("--batch_size_val", type=int, default=5, help="验证集的batch_size")
    parser.add_argument("--encoder_init_lr", type=float, default=3e-4, help="CNN特征编码器的初始lr")
    parser.add_argument("--decoder_init_lr", type=float, default=3e-4, help="RNN特征编码器的初始lr")
    # model config
    parser.add_argument("--att_dim", type=int, default=512, help="模型参数见model.py")
    parser.add_argument("--decoder_dim", type=int, default=512, help="模型参数见model.py")
    parser.add_argument("--embed_dim", type=int, default=512, help="模型参数见model.py")
    parser.add_argument("--continue_model", help="要继续训练的模型文件路径")
    parser.add_argument("--use_fusion", type=bool, default=True, help="融合多帧")
    arg = parser.parse_args()
    return arg


def main(continue_model=None):
    global vocab
    arg = getArg()
    vocab_path = arg.vocab_path
    vocab = Vocabulary()
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    print("vocab_size:", len(vocab))
    image_root_train = arg.image_root_train
    image_root_val = arg.image_root_val
    caption_path = arg.caption_path
    epochs = arg.epochs
    batch_size_train = arg.batch_size_train
    batch_size_val = arg.batch_size_val
    transforms = Compose([Resize((224, 224)),
                          RandomHorizontalFlip(),
                          RandomHorizontalFlip(),
                          RandomRotation(degrees=30),
                          ToTensor()])
    # Normalize([0.43710339, 0.41183448, 0.39289876],
    #           [0.27540463, 0.27135348, 0.27471914])])
    encoder_init_lr = arg.encoder_init_lr
    decoder_init_lr = arg.decoder_init_lr
    train_loader = get_loader(vocab_path, image_root_train,
                              caption_path, batch_size=batch_size_train,
                              transforms=transforms, use_fusion=arg.use_fusion)
    val_loader = get_loader(vocab_path, image_root_val,
                            caption_path, batch_size=batch_size_val,
                            transforms=transforms, use_fusion=arg.use_fusion)
    model_config = {"att_dim": arg.att_dim,
                    "decoder_dim": arg.decoder_dim,
                    "embed_dim": arg.embed_dim,
                    "vocab_size": len(vocab),
                    "encoder_dim": 2048}
    encoder = CNNEncoder(cnn_type="resnet")
    decoder = RNNDecoderWithAttention(**model_config)
    encoder_optimizer = torch.optim.Adam(params=encoder.parameters(), lr=encoder_init_lr)
    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=decoder_init_lr)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(loss_fn, train_loader, val_loader,
                      encoder_optimizer, decoder_optimizer,
                      encoder, decoder)
    trainer.check_init(clean_log=False, arg=arg)
    if arg.continue_model is not None:
        trainer.load_state(arg.continue_model, lr_list=[encoder_init_lr, decoder_init_lr])

    for epoch in range(epochs):
        trainer.train_loop()
        trainer.eval_loop(save_per_epochs=10)


if __name__ == '__main__':
    main()
