#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/13 下午8:16
# @Author : PH
# @Version：V 0.1
# @File : train.py
# @desc :
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToPILImage, ToTensor, \
    RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import pickle
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# 我自己的文件
from dataset import Airplane
from model import RCNN, LayerActivations
from utils.template import TemplateModel

# 超参数
BATCH_SIZE = 120
LEARNING_RATE = 3e-4
EPOCHS = 1
transforms = Compose([ToPILImage(),
                      # 概率水平翻转、垂直翻转、和随机旋转
                      RandomHorizontalFlip(p=0.5),
                      RandomVerticalFlip(p=0.5),
                      RandomRotation(degrees=60),
                      ToTensor()])
device = "cuda" if torch.cuda.is_available() else "cpu"

# 读取数据
dataset = Airplane("./region", transforms=transforms)
print(f"全部数据集数量：{len(dataset)}")
len_train = int(len(dataset) * 0.9)
len_test = int(len(dataset)) - len_train
print(f"训练集数量:{len_train} 测试集数量:{len_test}")
# 使用pytorch中random_split随机划分训练集和测试集
# 我试过sklearn中的train_test_split，会报错。原因可能是划分之前要将整个数据集中的图片读取到内存中导致内存不够
train_set, test_set = random_split(dataset, [len_train, len_test])
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

# model
model = RCNN()
model.to(device)
# optimizer
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
# loss_fn，因为模型最后经过fc层输出形状为（B，1）。且没有使用sigmoid函数激活。因此使用BCEWithLogitsLoss损失
loss_fn = nn.BCEWithLogitsLoss()


# metric
def metric(preds, gt):
    """

    Args:
        preds: tensor in cpu
        gt: tensor in cpu

    Returns:
        scores: dict
            各种性能指标的字典，这里只计算了准确率

    """
    scores = {}
    gt = gt.unsqueeze(dim=1).to(dtype=torch.float)
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    acc = (preds == gt).sum() / preds.shape[0]
    scores["acc"] = acc
    return scores


# 我自己的一个训练摸板，继承后修改__init__即可
class Trainer(TemplateModel):
    def __init__(self):
        super(Trainer, self).__init__()
        # check_point 目录
        self.ckpt_dir = "./check_point"
        # tensorboard
        self.writer = SummaryWriter()
        # 训练状态
        self.step = 0
        self.epoch = 0
        self.best_acc = 0.0
        # 模型架构
        self.model = model
        self.optimizer = optimizer
        self.criterion = loss_fn
        self.metric = metric
        # 数据集
        self.train_loader = train_loader
        self.test_loader = test_loader
        # 运行设备
        self.device = device
        # 训练时print和记录loss的间隔
        self.log_per_step = 50


def main():
    # 如果需要继续训练，model_path为需要继续训练的model路径
    model_path = None
    trainer = Trainer()
    trainer.check_init()
    if model_path:
        # 如果继续训练，恢复模型
        trainer.load_state(model_path)

    # 开始训练CNN
    for epoch in range(EPOCHS):
        print(20 * "*", f"epoch:{epoch + 1}", 20 * "*")
        trainer.train_loop()
        trainer.eval()

    # 训练svm分类器,先训练好cnn后.再使用最好的cnn训练svm分类器即可
    print("训练svm")
    inp_4096 = []
    ys = []
    # LayerActivations是基于torch中hook机制设计的一个类
    # 用于在model的某一层次的设置一个标记，以便获取该层次前向传播后的输出
    # 即获取中间层的输出
    feat_4096 = LayerActivations(trainer.model.new_layers, 0)  # 在model的最后的fc层中的第一个module设置hook
    # 用sklearn中SVC实现svm分类器，先标准化。这里使用默认的参数。并没有去调参
    svm_cls = Pipeline([("norm", StandardScaler()),
                        ("svm_cls", SVC())
                        ])
    trainer.load_state("./check_point/best.pth")
    for X, y in train_loader:
        # hook的机制就是模型前向传播处理input时，记录hook所在module的输出
        trainer.inference(X)  # 前向传播
        inp_4096.append(feat_4096.features.cpu())  # 此时feat_4096.features中就捕获到了中间层的输出
        ys.append(y.cpu())
    feat_4096.remove()  # 移除hook，我不知道不移除有啥后果....
    inp_4096 = torch.cat(inp_4096, dim=0)
    ys = torch.cat(ys, dim=0)
    # 使用SVC（）的fit训练svm分类器，这里inp_4096shape(36201, 4096)
    # 因此训练会非常慢(我的为10min)，还有可能内存不够...
    svm_cls.fit(inp_4096.detach().numpy(), ys.detach().numpy())
    # 用pickle保存svm模型
    with open("./check_point/svm_cls.pkl", "wb") as f:
        pickle.dump(svm_cls, f)
        print("保存svm_cls:./check_point/svm_cls.pkl")
        f.close()


if __name__ == '__main__':
    main()
