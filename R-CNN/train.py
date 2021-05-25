#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/13 下午8:16
# @Author : PH
# @Version：V 0.1
# @File : train.py
# @desc :
import time
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToPILImage, ToTensor, \
    RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import pickle
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# 我自己的文件
from dataset import Airplane
from model import RCNN, LayerActivations
from utils.template import TemplateModel

# 超参数
BATCH_SIZE = 120
LEARNING_RATE = 3e-4
EPOCHS = 10
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


def get4096Vec(trainer, data_loader):
    """
    提取CNN网络的fc层中第一个layer输出的长度为4096的特征向量

    Returns:
        inp_4096: torch.tensor shape"B, 4096"
            用于训练SVM分类器的特征向量
        ys：torch.tensor shape"B, 1"
            用于训练SVM分类器的特征向量对应的标签
    """
    inp_4096 = []
    ys = []
    # LayerActivations是基于torch中hook机制设计的一个类
    # 用于在model的某一层次设置一个标记，以便获取该层次前向传播后的输出
    # 即获取中间层的输出
    feat_4096 = LayerActivations(trainer.model.new_layers, 0)  # 在model的最后的fc层中的第一个module设置hook
    trainer.load_state("./check_point/best.pth")
    for X, y in data_loader:
        # hook的机制就是模型前向传播处理input时，记录hook所在module的输出
        trainer.inference(X)  # 前向传播
        inp_4096.append(feat_4096.features.cpu())  # 此时feat_4096.features中就捕获到了中间层的输出
        ys.append(y.cpu())
    feat_4096.remove()  # 移除hook，我不知道不移除有啥后果....
    inp_4096 = torch.cat(inp_4096, dim=0)
    ys = torch.cat(ys, dim=0)
    return inp_4096, ys


def trainSvm(trainer, evalSVM=True):

    # 提取CNN中间层输出的4096特征向量
    inp_4096, ys = get4096Vec(trainer, train_loader)

    # 用sklearn实现svm分类器，其中先标准化。
    # 使用过SVC默认参数，SVC(C=20), SVC(C=0.5) 以及 LinearSVC().最后选择了SVC()
    # LinearSVC最大迭代次数20000也无法收敛，这个没办法了只能用带‘rbf’核的SVC
    svm_cls = Pipeline([("norm", StandardScaler()),
                        ("svm_cls", SVC(verbose=True))
                        ])

    # 使用SVC（）的fit训练svm分类器，这里inp_4096 shape(36201, 4096)
    # 因此训练会非常慢(我的为10 - 30min)，还有可能内存不够...
    svm_cls.fit(inp_4096.detach().numpy(), ys.detach().numpy())

    # 评估模型
    if evalSVM:
        labels = [0, 1]
        target_names = ["background", "Airplane"]
        inp_4096, ys = get4096Vec(trainer, test_loader)
        preds = svm_cls.predict(inp_4096.detach().numpy())
        print(metrics.classification_report(preds, ys.detach().numpy(),
                                            target_names=target_names, labels=labels))

    # 用pickle保存svm模型
    with open("./check_point/svm_cls.pkl", "wb") as f:
        pickle.dump(svm_cls, f)
    print("保存svm_cls:./check_point/svm_cls.pkl")
    f.close()


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


def main(train_CNN=True, train_svm=True):
    start = time.time()
    # 如果需要继续训练，model_path为需要继续训练的model路径
    model_path = None  # "./check_point/best.pth"
    trainer = Trainer()
    trainer.check_init()
    if model_path:
        # 如果继续训练，恢复模型
        trainer.load_state(model_path)

    if train_CNN:
        # 开始训练CNN
        for epoch in range(EPOCHS):
            print(20 * "*", f"epoch:{epoch + 1}", 20 * "*")
            trainer.train_loop()
            trainer.eval(save_per_eval=True)

    # 训练svm分类器,先训练好cnn后.再使用最好的cnn训练svm分类器即可
    if train_svm:
        print("训练svm")
        trainSvm(trainer, evalSVM=True)
    t = time.time() - start
    print(f"Done!\tTotal time:{t}")


if __name__ == '__main__':
    main(train_CNN=True, train_svm=True)
