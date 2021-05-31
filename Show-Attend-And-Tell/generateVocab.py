#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/31 下午1:56
# @Author : PH
# @Version：V 0.1
# @File : generateVocab.py
# @desc : 将ground truth中的字符制作为字典vocabulary
import nltk
import collections
import json
import pickle


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def addWord(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def buildVocab(original_text_path="./video/video_demo/demo.json", threshold=0):
    # 用一个hash表来统计原始语料库中每个字符出现的次数
    counter = collections.Counter()
    #
    with open(original_text_path) as f:
        text_dicts = json.load(f)["sentences"]
        for i, text_dict in enumerate(text_dicts):
            tokens = nltk.word_tokenize(str(text_dict["caption"]).lower())
            counter.update(tokens)
            print("\r", f"已tokenize:[{i + 1}]/[{len(text_dicts)}]", end=" ")
        print("\n")
        # 只有语料库中出现次数超过阈值的字符才被记录到字典中
        words = [word for word, cnt in counter.items() if cnt > threshold]
        # 几个基础的特殊符号
        vocab = Vocabulary()
        vocab.addWord("<pad>")
        vocab.addWord("<start>")
        vocab.addWord("<end>")
        vocab.addWord("<unk>")
        # 根据words建立vocabulary
        for i, word in enumerate(words):
            vocab.addWord(word)
            print("\r", f"建立Vocab:[{i + 1}]/[{len(words)}]", end=' ')
        return vocab


if __name__ == "__main__":
    vocab = buildVocab()
    with open("./video/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
        print("保存vocab：./video/vocab.pkl")
    # word2idx = vocab.word2idx
    # for key, value in word2idx.items():
    #     print(key, " ", value)
