#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/14 下午4:59
# @Author : PH
# @Version：V 0.1
# @File : template.py
# @desc :
import torch
import os
import os.path as osp


class TemplateModel:
    def __init__(self):
        # tensorboard
        self.writer = None
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_acc = 0.0
        # 模型架构
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.metric = None  # 可没有
        # 数据集
        self.train_loader = None
        self.test_loader = None
        # 运行设备
        self.device = None
        # check_point 目录
        self.ckpt_dir = None
        # 训练时print的间隔
        self.log_per_step = None

    def check_init(self):
        assert self.model
        assert self.optimizer
        assert self.criterion
        assert self.metric
        assert self.train_loader
        assert self.test_loader
        assert self.device
        assert self.ckpt_dir
        assert self.log_per_step
        torch.cuda.empty_cache()
        if not osp.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)

    def load_state(self, fname, optim=True):
        state = torch.load(fname)

        if isinstance(self.model, torch.nn.DataParallel):  # 多卡训练
            self.model.module.load_state_dict(state['model'])
        else:  # 非多卡训练
            self.model.load_state_dict(state['model'])
        # 恢复一些状态参数
        if optim and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
        self.global_step = state['global_step']
        self.epoch = state['epoch']
        self.best_acc = state['best_acc']
        print('load model from {}'.format(fname))

    def save_state(self, fname, optim=True):
        state = {}

        if isinstance(self.model, torch.nn.DataParallel):
            state['model'] = self.model.module.state_dict()
        else:
            state['model'] = self.model.state_dict()
        # 除了保存模型的参数外，还要保存当前训练的状态：optim中的参数、epoch、step等
        if optim:
            state['optimizer'] = self.optimizer.state_dict()
        state['global_step'] = self.global_step
        state['epoch'] = self.epoch
        state['best_acc'] = self.best_acc
        torch.save(state, fname)
        print('save model at {}'.format(fname))

    def train_loop(self):
        self.model.train()
        self.epoch += 1
        running_loss = 0.0
        for step, batch in enumerate(self.train_loader):
            self.global_step += 1
            self.optimizer.zero_grad()
            loss = self.train_loss(batch)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if step % self.log_per_step == 0:
                # 记录每一批loss的平均loss
                avg_loss = running_loss / (self.log_per_step * len(batch))
                self.writer.add_scalar('loss', avg_loss, self.global_step)
                print(
                    f"loss:{avg_loss : .5f}\tcur:[{(step + 1) * self.train_loader.batch_size}]\[{len(self.train_loader.dataset)}]")

                # 记录参数和梯度
                for tag, value in self.model.named_parameters():
                    tag = tag.replace('.', '/')
                    self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy())
                    if value.grad is not None:  # 在FineTurn时有些参数被冻结了，没有梯度。也就不用记录了
                        self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy())

                running_loss = 0.0

    def train_loss(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.unsqueeze(dim=1).to(self.device, dtype=torch.float)

        pred = self.model(x)
        loss = self.criterion(pred, y)
        return loss

    def eval(self, save_per_eval=True):
        self.model.eval()
        # 如果要使用一些其他的性能指标，就要设置self.metric成员。然后返回一个有关指标的字典
        # 比如使用sklearn中的f1_score, recall...
        scores = self.eval_scores()
        for key in scores.keys():
            self.writer.add_scalar(f"{key}", scores[key].item(), self.epoch)
        if scores["acc"] >= self.best_acc:
            self.best_acc = scores["acc"]
            self.save_state(osp.join(self.ckpt_dir, f'best.pth'), False)
        if save_per_eval:  # 每次评估都保存当前模型？
            self.save_state(osp.join(self.ckpt_dir, f'epoch{self.epoch}.pth'))
        print('epoch:{}\tACC {:.5f}'.format(self.epoch, scores["acc"]))
        return scores["acc"]

    def eval_scores(self):
        xs, ys, preds = [], [], []
        # temp = {}
        for batch in self.test_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.model(x)

            # 下面的注释可以分批获取metric，但是要改metric()的实现
            # scores = self.metric(pred.cpu(), y.cpu())
            # for key in scores.keys():
            #     temp[key] += scores[key]

            xs.append(x.cpu())
            ys.append(y.cpu())
            preds.append(pred.cpu())
        # 将所有pred和label全部获取后才送入metric()计算性能指标的方法要小心内存不够...
        # 分批计算metric的方法要改metric的实现，目前还不想改...
        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        preds = torch.cat(preds, dim=0)
        scores = self.metric(preds, ys)
        return scores

    def inference(self, x):
        x = x.to(self.device)
        return self.model(x)

    def num_parameters(self):
        return sum([p.data.nelement() for p in self.model.parameters()])
