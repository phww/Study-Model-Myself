#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/14 下午8:47
# @Author : PH
# @Version：V 0.1
# @File : predict.py
# @desc :
import pickle

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

from model import RCNN, LayerActivations
import cv2
from cv2.ximgproc import segmentation
from torchvision.transforms import ToTensor, Resize, Compose
from utils.template import TemplateModel
from utils.nms import nms, nms2

cv2.setUseOptimized(True)
cv2.setNumThreads(8)
transforms = Compose([ToTensor(),
                      Resize((225, 225))])


def selective_search(img):
    ss = segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchQuality()  # Fast or Quality
    rects = ss.process()
    return rects


def generateBbox(img, rects, transforms, model, svm):
    # 使用训练好的CNN模型，从selective search算法中生成的候选区域中
    # 找出得分大于0.7的候选区域作为，包含检测目标的bbox
    bbox = []
    scores = []
    n = rects.shape[0] if rects.shape[0] <= 2000 else 2000  # 只考虑前2000个regions
    cnt = 0
    feat_4096 = LayerActivations(model.new_layers, 0)
    for i in tqdm(range(n), ncols=100, position=0, leave=True):
        x, y, w, h = rects[i]
        region = img[y:y + h, x:x + w]
        region = transforms(region)
        if svm is None:
            pred = model(region.unsqueeze(dim=0))
            pred = torch.sigmoid(pred)
            if pred.item() > 0.7:
                cnt += 1
                tqdm.write(f"find:{cnt}!!, {x}, {y}, {x + w}, {y + h}")
                bbox.append([x, y, x + w, y + h])
                scores.append(pred.item())

        else:
            score = model(region.unsqueeze(dim=0))
            score = torch.sigmoid(score)
            pred = svm.predict(feat_4096.features.detach().numpy())
            if pred == 1:
                cnt += 1
                tqdm.write(f"find:{cnt}!!, {x}, {y}, {x + w}, {y + h}")
                bbox.append([x, y, x + w, y + h])
                scores.append(score.item())

    print("送入NMS的bbox数量:", len(bbox))
    select_bbox = nms2(bbox, scores)
    print("NMS后的bbox数量:", len(select_bbox))
    feat_4096.remove()
    return select_bbox


def drawBbox(img, final_bbox):
    for i in range(len(final_bbox)):
        x1, y1, x2, y2, score = final_bbox[i]
        # score = select_bbox[i][1]
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(img, f"score:{score:.2f}", (x1, y1), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1)
    plt.imshow(img[..., ::-1])  # opencv 读取的图像为BGR，plt的为RGB。这里只是反转了颜色通道
    plt.show()


def main(img_path, model_path, svm):
    model = RCNN()
    state = torch.load(model_path)
    model.load_state_dict(state['model'])
    print(f"Best ACC:{state['best_acc']}")
    img = cv2.imread(img_path)
    rects = selective_search(img)
    print("regions:", len(rects))
    final_bbox = generateBbox(img, rects, transforms, model, svm)
    img_out = img.copy()
    drawBbox(img_out, final_bbox)


if __name__ == "__main__":
    img_path = "./test1.jpg"
    model_path = "./check_point/best.pth"
    svm_path = "./check_point/svm_cls.pkl"
    with open(svm_path, "rb") as f:
        svm_cls = pickle.load(f)
        f.close()
    main(img_path, model_path, svm_cls)
