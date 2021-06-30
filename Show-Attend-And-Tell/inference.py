#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/6/7 下午8:50
# @Author : PH
# @Version：V 0.1
# @File : inference.py
# @desc : 预测并将结果保存在json文件内
import json
import re
import os
import pickle
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from train import translate2Sentence
from model import CNNEncoder, RNNDecoderWithAttention
from generateVocab import Vocabulary
from tqdm.auto import tqdm
import pprint

global encoder, decoder
transforms = Compose([Resize((224, 224)),
                      ToTensor()])
# Normalize([0.43710339, 0.41183448, 0.39289876],
#           [0.27540463, 0.27135348, 0.27471914])])
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_path = "/home/ph/Dataset/VideoCaption/vocab.pkl"
vocab = Vocabulary()
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)


def getModel(model_t_path):
    model_config = {"att_dim": 512,
                    "decoder_dim": 512,
                    "embed_dim": 512,
                    "vocab_size": vocab_size,
                    "encoder_dim": 2048}
    encoder = CNNEncoder(cnn_type="resnet")
    decoder = RNNDecoderWithAttention(**model_config)
    encoder.load_state_dict(torch.load(model_t_path)["model0"])
    decoder.load_state_dict(torch.load(model_t_path)["model1"])
    return encoder, decoder


def inferenceOneImage(imgs, bean_width=5):
    k = bean_width
    with torch.no_grad():
        # encoder提取特征，并预处理
        feat = encoder(imgs)  # 1, 14, 14, 2048
        enc_image_size = feat.size(1)
        feat_dim = feat.size(-1)
        feat_tokens = feat.view(1, -1, feat_dim)  # 1, 14 x 14, 2048
        n_tokens = feat_tokens.size(1)
        # 使用beam search对一张图片同时生成k句描述
        feat_tokens = feat_tokens.expand(k, n_tokens, feat_dim)  # k, 196, 2048

        # 每个语句的开始的token为"<start>"，此时所有句子只含有<start>
        k_prev_words = torch.LongTensor([[vocab.word2idx["<start>"]]] * k).to(device)  # k, 1

        # 保存每个句子，目前和k_prev_words一样
        seqs = k_prev_words
        # 保存top k 语句的得分
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
        # 前k个语句的attend权重矩阵,数值范围[0，1]. (k, 1, enc_image_size, enc_image_size)
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)

        # 保存完整推断的句子，attend权重和得分
        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # 开始decoding
        step = 1
        # 初始的h0，c0
        h, c = decoder.init_hidden_state(feat_tokens)
        while True:
            # 打印信息
            # print(f"step{step}")
            # print("incomplete")
            # v = 1.0
            # for i, seq in enumerate(seqs):
            #     v += top_k_scores[i].item()
            #     print(f"{vocab.idx2word[seq[-1].item()]}:{v}")
            # for i, seq in enumerate(seqs):
            #     print(translate2Sentence(words_vec=[seq.cpu().numpy()], vocab=vocab, reference=False)[0])
            # print("complete")
            # for seq in complete_seqs:
            #     print(translate2Sentence(words_vec=[seq], vocab=vocab, reference=False)[0])

            embeddings = decoder.embedding(k_prev_words)  # k, 1, embed_dim
            # s，2048 和 s，enc_image_size**2
            att_weighted_encoder_out, att_weight = decoder.attention(feat_tokens, h)
            alpha = att_weight.view(-1, enc_image_size, enc_image_size)  # s, enc_image_size, enc_image_size
            gate = decoder.sigmoid(decoder.f_beta(h))
            att_weighted_encoder_out *= gate
            h, c = decoder.lstm_pre_step(torch.cat([embeddings.squeeze(dim=1), att_weighted_encoder_out], dim=1),
                                         (h, c))  # s, decoder_dim

            preds = decoder.fc(decoder.dropout(h))
            scores = F.log_softmax(preds, dim=1)  # s, vocab_size 数值范围【0，1】
            scores = top_k_scores.expand_as(scores) + scores  # s, vocab_size

            if step == 1:
                # 刚开始时每个句子当前的词都是<start>，因此只需要<start>这一个词参与下个词的推断
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # s
            else:
                # 其他时刻，要依据上一时刻推断的k个词，推断该时刻的前k个词
                # 注意这里的前k个词需要从整体k个语句上考虑，而不是只考虑一个语句
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # s

            #
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # 将该时刻推断出来的词和attend权重保存
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            # (s, step+1, enc_image_size, enc_image_size)
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)

            # 如果当前推断的词不是<end>代表当前句子的推断没有完成
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != vocab.word2idx['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # 保存推断完成的句子，即推断到<end>的句子
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            # 推断完成一个句子，不要忘记使k值减小
            k -= len(complete_inds)

            # 直到推断完k个句子，退出算法
            if k == 0:
                break

            # 更新未推断完成的句子的信息，参与到下一次的迭代
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            feat_tokens = feat_tokens[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # 如果一个句子一直推断不到<end>,就要设置一个最大推断长度让算法退出循环
            if step > 50:
                break
            step += 1
        # beam search完成，得到了k个描述性语句。选择其中得分最高的语句，作为最终输出
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]
        return seq, alphas


def main(img_root, model_path, bean_width=5, k=8):
    global encoder, decoder
    encoder, decoder = getModel(model_path)
    encoder.to(device).eval()
    decoder.to(device).eval()
    img_paths = os.listdir(img_root)
    img_paths = sorted(img_paths)
    save = {"predictions": {}}
    f = open("./answer.json", "w", encoding='utf-8')
    for i in tqdm(range(len(img_paths) // k)):
        img_path = img_paths[i * k]
        # 从G-1000_8.jpg -> G-1000_8 -> G-1000
        video_id = img_path.split(".")[0]
        video_id = video_id.split("-")[0]
        imgs = Image.open(os.path.join(img_root, img_path)).convert('RGB')
        imgs = transforms(imgs).unsqueeze(dim=0)  # 1, 3, 224, 224
        for j in range(1, k):
            img_path = img_paths[i * k + j]
            img = Image.open(os.path.join(img_root, img_path)).convert('RGB')
            img = transforms(img).unsqueeze(dim=0)
            imgs = torch.cat([imgs, img], dim=1)  # 1, 3*k, 224, 224
        imgs = imgs.to(device)

        seq, alphas = inferenceOneImage(imgs, bean_width=bean_width)
        caption = translate2Sentence(words_vec=[seq], vocab=vocab, reference=False)[0]
        save_sentence = str()
        for word in caption:
            if word not in ["<start>", "<end>"]:
                save_sentence += word
                save_sentence += " "
        save_inner = [{"image_id": video_id, "caption": save_sentence}]
        print(save_sentence)
        if video_id not in save["predictions"].keys():
            save["predictions"][video_id] = save_inner
        elif len(save_sentence) > len(save["predictions"][video_id][0]["caption"]) and k == 1:
            save["predictions"][video_id] = save_inner

    json.dump(save, f, indent=4)


if __name__ == '__main__':
    main(img_root="/home/ph/Dataset/VideoCaption/generateImgs/test",
         model_path="./check_point5/best.pth", bean_width=100, k=1)  # 3060 bean_width=3000时爆显存了
