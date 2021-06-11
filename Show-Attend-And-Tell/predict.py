#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/6/7 下午8:50
# @Author : PH
# @Version：V 0.1
# @File : predict.py
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

global encoder, decoder
transforms = Compose([Resize((224, 224)),
                      ToTensor(),
                      Normalize([0.43710339, 0.41183448, 0.39289876],
                                [0.27540463, 0.27135348, 0.27471914])])
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_path = "/home/ph/Dataset/VideoCaption/vocab.pkl"
vocab = Vocabulary()
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)


def get_model(model_t_path):
    model_config = {"att_dim": 512,
                    "decoder_dim": 512,
                    "embed_dim": 512,
                    "vocab_size": vocab_size}
    encoder = CNNEncoder()
    decoder = RNNDecoderWithAttention(**model_config)
    encoder.load_state_dict(torch.load(model_t_path)["model0"])
    decoder.load_state_dict(torch.load(model_t_path)["model1"])
    return encoder, decoder


def inferenceOneImage(img_path, bean_width=5):
    k = bean_width
    with torch.no_grad():
        # 处理image
        img = Image.open(img_path).convert('RGB')
        img = transforms(img).unsqueeze(dim=0)  # 1, 3, 256, 256
        img = img.to(device)

        # encoder提取特征，并预处理
        feat = encoder(img)  # 1, 14, 14, 2048
        enc_image_size = feat.size(1)
        feat_dim = feat.size(-1)
        feat_tokens = feat.view(1, -1, feat_dim)  # 1, 14 x 14, 2048
        n_tokens = feat_tokens.size(1)
        # 对一张图片同时生成k句描述
        feat_tokens = feat_tokens.expand(k, n_tokens, feat_dim)  # k, 196, 2048

        # 生成top k 的语句
        # 每个语句的开始的token,"<start>"
        k_prev_words = torch.LongTensor([[vocab.word2idx["<start>"]]] * k).to(device)  # k, 1
        seqs = k_prev_words
        # 保存top k 语句的得分
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
        # Tensor to store top k sequences' alphas; now they're just 1s
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(
            device)  # (k, 1, enc_image_size, enc_image_size)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # 开始decoding
        step = 1
        h, c = decoder.init_hidden_state(feat_tokens)
        while True:
            embeddings = decoder.embedding(k_prev_words)  # k, 1,
            att_weighted_encoder_out, att_weight = decoder.attention(feat_tokens, h)
            alpha = att_weight.view(-1, enc_image_size, enc_image_size)
            gate = decoder.sigmoid(decoder.f_beta(h))
            att_weighted_encoder_out *= gate
            h, c = decoder.lstm_pre_step(
                torch.cat([embeddings.squeeze(dim=1), att_weighted_encoder_out], dim=1), (h, c))

            preds = decoder.fc(decoder.dropout(h))  # s, vocab_size
            scores = F.log_softmax(preds, dim=1)
            scores = top_k_scores.expand_as(scores) + scores  # s, vocab_size

            if step == 1:
                # 刚开始时每个句子当前的词都是<start>，因此只需要<start>预测的前k个词
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # s
            else:
                # 其他时刻会预测k个词，要基于这k个词找预测结果
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # s

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences, alphas
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                   dim=1)  # (s, step+1, enc_image_size, enc_image_size)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != vocab.word2idx['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            feat_tokens = feat_tokens[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]
        return seq, alphas


def main(img_root, model_path, bean_width=5):
    global encoder, decoder
    encoder, decoder = get_model(model_path)
    encoder.to(device).eval()
    decoder.to(device).eval()
    img_paths = os.listdir(img_root)
    save = {"predictions": {}}
    f = open("./answer.json", "w", encoding='utf-8')
    for img_path in tqdm(img_paths):
        img_name = img_path.split(".")[0]
        img_name = img_name.split("-")[0]

        seq, alphas = inferenceOneImage(os.path.join(img_root, img_path),
                                        bean_width=bean_width)
        caption = translate2Sentence(words_vec=[seq], vocab=vocab, reference=False)[0]
        # print(caption)
        save_sentence = str()
        for word in caption:
            if word not in ["<start>", "<end>"]:
                save_sentence += word
                save_sentence += " "
        save_inner = [{"image_id": img_name, "caption": save_sentence}]
        if img_name not in save["predictions"].keys():
            save["predictions"][img_name] = save_inner

    json.dump(save, f, indent=4)


if __name__ == '__main__':
    main(img_root="/home/ph/Dataset/VideoCaption/generateImgs/test",
         model_path="./check_point5/epoch100.pth", bean_width=100)  # 3060 bean_width=3000时爆显存了
