#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/14 下午8:46
# @Author : PH
# @Version：V 0.1
# @File : model_my.py
# @desc :VIT
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

device = "cuda" if torch.cuda.is_available() else "cpu"


class EmbedPatch(nn.Module):
    """将图片拆解为多个patches，并嵌入到指定的维度中去"""

    def __init__(self, img_size, patch_size, in_channels, emb_channels):
        """

        Args:
            img_size: int
                输入图片的大小
            patch_size: int
              每个patch的大小
            in_channels: int
                输入图片的通道数
            emb_channels: int
                嵌入向量的长度
        Attributes：
            self.num_patches: int
                将一副图片拆解为patches的个数,即n_tokens
            self.proj: nn.Conv2d
                将原始图像，转换为 shape "batch_size, emb_channels, num_patches**0.5, num_patches**0.5"
                表示原始图片变成了一个num_patches**0.5 X num_patches**0.5的矩阵，其中每个矩阵就对应一个emb_patches

        """
        super(EmbedPatch, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels,
                              emb_channels,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, x):
        """

        Args:
            x: torch.tensor
                x.shape "batch_size, in_channels, img_size, img_size"

        Returns:
            patches: torch.tensor
                patch.shape "batch_size, num_patches, emb_channels"

        """
        patches = self.proj(x)  # patches.shape=(batch_size, emd_channels, n_patch**0.5, patch_size**0.5)
        # patches = patches.flatten(2)  # patches.shape=(batch_size, emb_channels, num_patches)
        # patches = patches.transpose(1, 2)  # patches.shape=(batch_size, num_patches, emb_channels)
        patches = rearrange(patches, "b e ph pw-> b (ph pw) e")
        return patches


class MultiHeadSelfAttention(nn.Module):
    """MSA的实现"""

    def __init__(self, n_head, n_dim, use_bias, dropout):
        """

        Args:
            n_head:int
                多头注意力的头数
            n_dim:int
                嵌入向量的长度，即emb_channels
            use_bias: bool
                qkv的线性变换中是否加上偏执项
            dropout:float

        Attributes:
            self.qkv: nn.Linear
                将输入转换为q，k，v的线性变换函数。这里直接将输入的最后一个维度变成3倍，之后在分开给q，k，v
            self.d_qkv: int
                多头注意力下q，k，v的嵌入维度的长度，即n_dim // n_head

        """
        super(MultiHeadSelfAttention, self).__init__()
        self.n_head = n_head
        self.n_dim = n_dim
        self.qkv = nn.Linear(n_dim, 3 * n_dim, bias=use_bias)
        self.d_qkv = n_dim // n_head
        self.scale = self.d_qkv ** 0.5
        if n_dim % n_head != 0:
            raise ValueError
        self.proj = nn.Linear(n_dim, n_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        """

        Args:
            x: torch.tensor
                输入是已经嵌入了的数据 shape"batch_size, n_tokens, n_dim"
        Returns:
            output: torch.tensor
                shape "batch_size, n_tokens, n_dim"
            attn: torch.tensor
                注意力矩阵 shape"batch_size, n_head, n_tokens, n_tokens"
        """
        batch_size, n_tokens, emb_channels = x.shape
        if emb_channels != self.n_dim:
            raise ValueError
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b n (e1 h e) ->e1 b h n e",
                        e1=3, h=self.n_head)  # shape"3, batch_size, n_head, n_tokens, n_dim"
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v = torch.chunk(qkv, 3, dim=-1)  # shape"batch_size, n_tokens, emb_channels"
        # q = q.reshape(
        #     batch_size, n_tokens, self.n_head, self.d_qkv).transpose(1, 2)
        # #shape"batch_size, n_tokens, n_head, n_dim"
        # k = k.reshape(batch_size, n_tokens, self.n_head, self.d_qkv).transpose(1, 2)
        # v = v.reshape(batch_size, n_tokens, self.n_head, self.d_qkv).transpose(1, 2)
        k_t = k.transpose(-2, -1)
        dp = q @ k_t  # shape "batch_size, n_head, n_tokens, n_tokens"
        attn = F.softmax(dp, dim=-1)
        attn = self.attn_drop(attn)
        weight_avg = attn @ v  # shape"batch_size, n_head, n_tokens, n_dim"
        output = self.attn_drop(weight_avg).transpose(1, 2)
        output = output.flatten(2)  # shape"batch_size, n_tokens, emb_channels"
        output = self.proj(output)
        output = self.proj_drop(output)
        return output, attn


class MLP(nn.Module):
    """接在MSA后面的线性变换（2层）"""

    def __init__(self, in_channels, n_hid, dropout):
        """

        Args:
            in_channels: int
                即n_dim
            n_hid: int
                MLP中间隐藏层的大小
            dropout: float
        Attributes：
            fc1: nn.Linear
            fc2: nn.Linear
            act: nn.GELU
                激活函数
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, n_hid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(n_hid, in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """

        Args:
            x:torch.tensor
                shape "batch_size, n_tokens, n_dim"

        Returns:
            x: torch.tensor
                shape "batch_size, n_tokens, n_dim"

        """
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """MSA + MLP + LayerNorm"""

    def __init__(self, n_head, n_dim, n_hid_ratio, use_bias, dropout):
        """

        Args:
            n_head: int
            n_dim: int
            n_hid_ratio: float[0,1]
                隐藏层大小比例，即n_hid = n_dim * n_hid_ratio
            use_bias: bool
            dropout: float

        Attributes:
            self.attn: MSA
            self.mlp: MLP
            self.norm: LayerNorm
        """
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(n_dim, 1e-6)
        self.attn = MultiHeadSelfAttention(n_head, n_dim, use_bias, dropout)
        self.norm2 = nn.LayerNorm(n_dim, 1e-6)
        self.mlp = MLP(in_channels=n_dim, n_hid=n_dim * n_hid_ratio, dropout=dropout)

    def forward(self, x):
        """

        Args:
            x:torch.tensor
                已经嵌入了的输入x shape "batch_size, n_tokens, n_dim"

        Returns:
            output:torch.tensor
                sequence to sequence的同形状变换 shape "batch_size, n_tokens, n_dim"
        """
        x = x + self.attn(self.norm1(x))[0]
        output = x + self.mlp(self.norm2(x))
        return output


class VisionTransformer(nn.Module):
    """VIT"""

    def __init__(self,
                 img_size,
                 patch_size,
                 n_dim,
                 n_classes,
                 img_channels=3,
                 n_depth=6,
                 n_head=8,
                 mlp_ratio=4,
                 use_bias=True,
                 dropout=0.1):
        """

        Args:
            img_size: int
            patch_size: int
            n_dim: int
            n_classes: int
                分类任务的分类数
            img_channels: default to 3
                图片通道数
            n_depth: default to 6
                block的层数
            n_head: default to 8
            mlp_ratio: default to 4
            use_bias: default to True
            dropout: default to 0.1

        Attributes:
            self.cls_emb: torch.tensor
                类别编码嵌入 shape"1, 1, n_dim",为了嵌入到每个输入x的头部中去，forward中会将0维度重复batch_size次
            self.pos_emb: torch.tensor
                位置编码嵌入 shape"1, 1 + n_patches, n_dim"，在forward中以广播的方式与输入x相加
            self.head: nn.Linear
                对n_dim进行降维，使维度和分类任务的分类数一样
        """
        super(VisionTransformer, self).__init__()
        self.embed_layer = EmbedPatch(img_size, patch_size, img_channels, emb_channels=n_dim)
        self.blocks = nn.ModuleList([Block(n_head=n_head,
                                           n_dim=n_dim,
                                           n_hid_ratio=mlp_ratio,
                                           use_bias=use_bias,
                                           dropout=dropout)
                                     for _ in range(n_depth)])
        self.cls_emb = nn.Parameter(torch.zeros((1, 1, n_dim)))
        self.pos_emb = nn.Parameter(torch.zeros((1, 1 + self.embed_layer.num_patches, n_dim)))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_dim, eps=1e-6)
        self.head = nn.Linear(n_dim, n_classes)

    def forward(self, x):
        """

        Args:
            x : torch.tensor
                shape "batch_size, img_channels, img_size, img_size"

        Returns:
            output : torch.tensor
                shape "batch_size, n_classes"

        """
        batch_size = x.shape[0]
        x = self.embed_layer(x)
        cls_emb = einops.repeat(self.cls_emb, "b p d ->(repeat b) p d", repeat=batch_size)
        # cls_emb = self.cls_emb.expand(batch_size, -1, -1)
        x = torch.cat((cls_emb, x), dim=1)
        x = self.dropout(x + self.pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]  # 最后的分类任务的特征向量，只要CLS部分的就行了。为什么只要CLS部分就可以用于分类任务了？？？
        output = self.head(cls_token_final)
        return output


if __name__ == "__main__":
    # test
    graphviz = GraphvizOutput()
    graphviz.output_file = 'basic.png'
    with PyCallGraph(output=graphviz):
        x = torch.randn(32, 3, 64, 64)
        # eb = EmbedPatch(64, 32, 3, 1000)
        # x = eb(x)
        # print(x.shape)
        # attention = MultiHeadSelfAttention(10, 1000, 0.1)
        # x, attn = attention(x)
        # print(x.shape, attn.shape)
        # mlp = MLP(1000, 512, 0.1)
        # x = mlp(x)
        # block = Block(10, 1000, n_hid_ratio=4, use_bias=True, dropout=0.1)
        # x = block(x)
        VIT = VisionTransformer(img_size=64,
                                patch_size=32,
                                n_dim=800,
                                n_classes=10)

        print(VIT)
        x = VIT(x)
        print(x.shape)
