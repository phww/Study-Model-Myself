#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/30 下午1:54
# @Author : PH
# @Version：V 0.1
# @File : model.py
# @desc : resnet101 + attention + LSTM
import torch
import torch.nn as nn
from torchvision.models import resnet101
from torchsummary import summary
from einops import rearrange


class CNNEncoder(nn.Module):
    def __init__(self, encode_image_size=14, fine_turn_layer=-3):
        super(CNNEncoder, self).__init__()
        resnet = resnet101(pretrained=True)
        # 不需要resnet101的最后两层
        modules = list(resnet.children())[:-2]
        self.extract_feat = nn.Sequential(*modules)
        # resnet倒数第二层的输出形状为（b，2048， 2， 2）需要变为（b，2048， 14， 14）
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(encode_image_size, encode_image_size))
        # fine turn最后3层，即特征向量长度为2048的卷积层
        self._fine_turn(fine_turn_layer)

    def forward(self, images):
        features = self.extract_feat(images)
        features_2048x14x14 = self.pooling(features)
        # 调整形状为(b, 14, 14, 2048)
        output = rearrange(features_2048x14x14, "B N H W -> B H W N")
        return output

    def _fine_turn(self, fin_turn_layer):
        for params in self.extract_feat.parameters():
            params.requires_grad = False
        for params in list(self.extract_feat.modules())[fin_turn_layer:]:
            params.requires_grad = True


class Attention(nn.Module):
    """注意力模块：通过CNN提取特征和RNN每个隐藏态的关系得到attention权重，为特征图的每个像素加权"""

    def __init__(self, att_dim, encoder_out_dim, rnn_hidden_dim):
        super(Attention, self).__init__()
        # 统一CNN输出、rnn隐藏态的特征向量的长度为att_dim
        self.encoder_att = nn.Linear(encoder_out_dim, att_dim)
        self.hidden_att = nn.Linear(rnn_hidden_dim, att_dim)
        # 将attention的特征维度降为1
        self.full_att = nn.Linear(att_dim, 1)
        # relu激活后使用softmax将attention权重归一化，这样权重之和为1。才符合权重的概念
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, rnn_hidden):
        """

        Args:
            encoder_out: CNN编码器提取的图像特征,shape"B, 14x14, 2048".
                这里将H和W维度压缩到了一起。可以理解为：特征图有14*14个像素，每个像素的特征向量的维度为2048
            rnn_hidden: RNN解码器的隐藏态，shape"B, 2048"

        Returns:
            encoder_out_weighted: 加了权的图像特征，shape"B, 2048"
            att_weight：权重矩阵，用于可视化网络进行推断时更关注原始图像的哪个部分（像素）,shape“B, 14*14”

        """
        encoder_att = self.encoder_att(encoder_out)
        # 下面要求和先扩充对应的维度
        rnn_hidden_att = self.hidden_att(rnn_hidden).unsqueeze(dim=1)
        # 经过full_att后输出形状为"B, 14x14, 1"，最后的1我们将其压缩掉
        att = self.full_att(self.relu(encoder_att + rnn_hidden_att)).squeeze(dim=2)
        # 归一化的attention才能称为权重，其shape"B, 14*14"
        att_weight = self.softmax(att)
        # 加权CNN提取的特征,因为encoder_out.shape=(B, 14*14, 2048),权重shape=(B, 14*14)
        # 因此权重要先在最后加一个维度，最终加权输出形状为(B, 2048)。刚好和RNN要求的隐藏态的形状一样！！
        encoder_out_weighted = (encoder_out * att_weight.unsqueeze(dim=2)).sum(dim=1)
        return encoder_out_weighted, att_weight


class RNNDecoderWithAttention(nn.Module):
    def __init__(self, att_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(RNNDecoderWithAttention, self).__init__()
        self.att_dim = att_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size
        # attention模块
        self.attention = Attention(att_dim, encoder_out_dim=encoder_dim, rnn_hidden_dim=decoder_dim)
        # 词嵌入模块
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        # lstm模块
        self.lstm_pre_step = nn.LSTMCell(input_size=encoder_dim + embed_dim, hidden_size=decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        # 初始化词嵌入层和全连接层的参数
        self._init_weights()

    def _init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def _init_hidden_state(self, encoder_out):
        """
        RNN初始状态的h0与c0定义为CNN模块输出的feature map
        在第二个维度上的平均值，并且用线性映射将特征向量的维
        度映射到嵌入向量的维度
        """
        mean = encoder_out.mean(dim=1)
        h0 = self.init_h(mean)
        c0 = self.init_c(mean)
        return h0, c0

    def forward(self, encoder_out, caption_gt, max_length=25):
        batch_size = encoder_out.size(0)
        pixels = encoder_out.size(1)
        # 词嵌入
        caption_gt = caption_gt.to(dtype=torch.long)
        caption_embed = self.embedding(caption_gt)
        # 展开特征图shape “B, 14, 14, 2048”->“B, 14*14, 2048”
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)
        # 初始的h0和c0
        h, c = self._init_hidden_state(encoder_out)

        # LSTM
        pred_word_vec = torch.zeros(batch_size, max_length, self.vocab_size)
        att_weights = torch.zeros(batch_size, max_length, pixels)

        for t in range(max_length):
            # 使用t-1时刻的隐藏态h与CNN提取的feature map计算attention。并使用这个att加权feature map
            # 这就是为什么可以可视化每个时刻，模型更关心feature map的哪个区域
            att_weighted_encoder_out, att_weight = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            att_weighted_encoder_out *= gate
            # 更新t时刻的h和"记忆细胞"c
            h, c = self.lstm_pre_step(torch.cat([caption_embed[:, t, :], att_weighted_encoder_out], dim=1), (h, c))
            # 用t时刻的h预测
            preds = self.fc(self.dropout(h))
            # 保存每个时刻的结果
            pred_word_vec[:, t, :] = preds
            att_weights[:, t, :] = att_weight
        return pred_word_vec, caption_gt, att_weights

cnn = CNNEncoder()
attention = Attention(100, 2048, 2048)
rnn_decoder = RNNDecoderWithAttention(att_dim=512, embed_dim=512, decoder_dim=512, vocab_size=100)

h = torch.randn(32, 2048)
# summary(cnn, (3, 512, 512), 32, "cpu")
# summary(attention, [(196, 2048), (2048,)], 32, "cpu")
summary(rnn_decoder, [(14 * 14, 2048), (25,)], 32, "cpu")
