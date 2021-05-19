#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/13 下午3:39
# @Author : PH
# @Version：V 0.1
# @File : model.py
# @desc : Fine Turn VGG16
import torch
from torchvision.models import vgg16_bn
import torch.nn as nn
from torchsummary import summary
from sklearn.svm import SVC


class LayerActivations:
    """
    参考别人的一个小方法：获取中间层的输出。具体用法见train.py
    """
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


class RCNN(nn.Module):
    """
    Fine Turn VGG_16作为特征提取网络
    """

    def __init__(self):
        super(RCNN, self).__init__()
        self.vgg_ft = vgg16_bn(pretrained=True)
        # 冻结vgg16的全部参数
        for params in self.vgg_ft.parameters():
            params.requires_grad = False
        # 替换vgg16的classifier层，使最终输出的特征向量长度为1
        self.new_layers = nn.Sequential(nn.Linear(25088, 4096),  # train.py中会在这里设置hook，以获取该层的输出用于训练svm
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(1000, 1))
        self.vgg_ft.classifier = self.new_layers

    def forward(self, inp):
        return self.vgg_ft(inp)


if __name__ == "__main__":
    # 测试
    inp = torch.rand(32, 3, 224, 224).cuda()
    label = torch.randint(0, 2, (32,))
    test = torch.rand(100, 4096)
    model = RCNN()
    feat4096 = LayerActivations(model.new_layers, 0)
    model.cuda()
    summary(model, (3, 224, 224), batch_size=32, device="cuda")
    svm_cls = SVC()
    feats = model(inp)
    print(feat4096.features.shape)
    print(feats.shape)
    svm_cls.fit(feat4096.features.cpu().detach().numpy(), label.cpu().detach().numpy())
    pred = svm_cls.predict(test.numpy())
    print(pred)
