#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/15 下午1:11
# @Author : PH
# @Version：V 0.1
# @File : nms.py
# @desc : 非极大值抑制
import numpy as np


def calc_ious(a, areas, x1, x2, y1, y2):
    area_a = np.abs(a[0] - a[2]) * np.abs(a[1] - a[3])
    # max_bbox 与其他bbox的所有交集的面积
    Ix1 = np.maximum(a[0], x1)
    Ix2 = np.minimum(a[2], x2)
    Iy1 = np.maximum(a[1], y1)
    Iy2 = np.minimum(a[3], y2)
    Iw = np.maximum(Ix2 - Ix1, 0)
    Ih = np.maximum(Iy2 - Iy1, 0)
    area_I = Iw * Ih
    area_U = area_a + areas - area_I
    return area_I * 1.0 / area_U

# 自己的实现，但是好像有小bug？？？
def nms(bbox, scores):
    bbox = np.array(bbox)
    scores = np.array(scores)
    sort_idx = np.argsort(-scores)  # 按照降序排序
    sort_bbox = bbox[sort_idx]
    sort_scores = scores[sort_idx]
    # 所有bbox的面积,calc_ious()会用到
    x1 = sort_bbox[:, 0]
    y1 = sort_bbox[:, 1]
    x2 = sort_bbox[:, 2]
    y2 = sort_bbox[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # 选择最终bbox
    select_bbox = []
    while len(sort_bbox) != 0:
        select_bbox.append((sort_bbox[0].tolist(), sort_scores[0]))  # 保存分数最大的bbox和分数
        # 对得分最大max_bbox计算其他other_bbox和它的iou，大于0.4的other_bbox全部从初始bbox array中删除（包括当前max_bbox）
        ious = calc_ious(sort_bbox[0], areas, x1, x2, y1, y2)
        # 删除和当前bbox的iou高于iou的其他bbox
        del_idx = np.where(ious > 0.4)  # 用where返回符合条件的idx
        sort_bbox = np.delete(sort_bbox, del_idx, axis=0)
        sort_scores = np.delete(sort_scores, del_idx)
        areas = np.delete(areas, del_idx)
        x1, x2 = np.delete(x1, del_idx), np.delete(x2, del_idx)
        y1, y2 = np.delete(y1, del_idx), np.delete(y2, del_idx)
    return select_bbox

# 网上别人的实现
def nms2(dets, score):
    dets = np.array(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    score = np.array(score)

    order = score.argsort()[::-1]
    area = (x2 - x1) * (y2 - y1)
    res = []
    while order.size >= 1:
        i = order[0]
        res.append([x1[i], y1[i], x2[i], y2[i], score[i]])

        # intersect area left top point(xx1, yy1): xx1 >= x1, yy1 >= y1
        # intersect area right down point(xx2, yy2): xx2 <= x2, yy2 <= y2

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, (xx2 - xx1))
        h = np.maximum(0.0, (yy2 - yy1))
        intersect = w * h

        # iou = intersect area / union; union = box1 + box2 - intersect
        iou = intersect / (area[i] + area[order[1:]] - intersect)

        # update order index;ind +1:because ind is obtain by index [1:]
        ind = np.where(iou <= 0.4)[0]
        order = order[ind + 1]

    return res
