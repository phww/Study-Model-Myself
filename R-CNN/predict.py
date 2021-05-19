#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/14 下午8:47
# @Author : PH
# @Version：V 0.1
# @File : predict.py
# @desc :
import os.path
import pickle
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
from model import RCNN, LayerActivations
import cv2
from cv2.ximgproc import segmentation
from torchvision.transforms import ToTensor, Resize, Compose
from utils.nms import nms, nms2

# cv2加速
cv2.setUseOptimized(True)
cv2.setNumThreads(8)
transforms = Compose([ToTensor(),
                      Resize((225, 225))])


def selectiveSearch(img):
    """
    基于cv2-python的selectiveSearch
    """
    ss = segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchQuality()  # Fast or Quality
    rects = ss.process()
    return rects


def generateBbox(img, rects, transforms, model, svm, info):
    """
    使用训练好的CNN模型，从selective search算法中生成的候选区域中找出包含目标的bbox
    Args:
        img: np.array
        rects: np.array shape “N,4”
            selective search算法生成的候选区域.其中每个区域由左上角坐标(x,y)以及bbox的(w,h)表示
        svm: 如果非None，使用svm分类器对候选区域进行预测。否则使用CNN网络最后的输出（shape"N, 1"），经过sigmoid处理后做预测
    Returns:
        select_bbox：list shape “N，4”
        经过非极大值抑制后的最终bbox
    """
    bbox = []
    scores = []
    n = rects.shape[0] if rects.shape[0] <= 2000 else 2000  # 最多只考虑前2000个regions
    cnt = 0
    feat_4096 = LayerActivations(model.new_layers, 0)
    for i in tqdm(range(n), ncols=100, position=0, leave=True):
        x, y, w, h = rects[i]
        # 切割出bbox中包含的图片
        region = img[y:y + h, x:x + w]
        region = transforms(region)
        # 不使用svm做分类预测，而是直接使用整个训练好的CNN做分类预测
        if svm is None:
            pred = model(region.unsqueeze(dim=0))
            pred = torch.sigmoid(pred)
            if pred.item() > 0.7:
                cnt += 1
                tqdm.write(f"find:{cnt}!!, {x}, {y}, {x + w}, {y + h}")
                bbox.append([x, y, x + w, y + h])
                scores.append(pred.item())

        # 使用SVM做分类预测
        else:
            score = model(region.unsqueeze(dim=0))
            score = torch.sigmoid(score)
            # if score.item() > 0.7:
            pred = svm.predict(feat_4096.features.detach().numpy())
            if pred == 1:
                cnt += 1
                tqdm.write(f"find:{cnt}!!, {x}, {y}, {x + w}, {y + h}")
                bbox.append([x, y, x + w, y + h])
                scores.append(score.item())
    # 非极大值抑制NMS
    print("送入NMS的bbox数量:", len(bbox))
    select_bbox = nms2(bbox, scores)
    print("NMS后的bbox数量:", len(select_bbox))
    feat_4096.remove()
    # info
    info.append(len(bbox))
    info.append(len(select_bbox))
    if svm is not None:
        info.append("svm")
    else:
        info.append("no-svm")
    return select_bbox, info


def drawBbox(img, final_bbox, info, save_path='./img_pred'):
    """
    绘制在预测的图片上绘制目标的bbox，并保存图片
    """
    for i in range(len(final_bbox)):
        x1, y1, x2, y2, score = final_bbox[i]
        # score = select_bbox[i][1]
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(img, f"score:{score:.2f}", (x1, y1), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[..., ::-1])  # opencv 读取的图像为BGR，plt的为RGB。这里只是反转了颜色通道
    plt.show()
    if save_path is not None:
        # 图片名为：ss算法生成的候选区域数量-网络最终分为正列的数量-NMS处理后的bbox数量-使用svm？-图片名称
        msg = f"{info[1]}-{info[2]}-{info[3]}-{info[4]}-{info[0]}"
        img_path = os.path.join(save_path, msg)
        plt.imsave(img_path, img[..., ::-1])
        print(f"保存图片：{img_path}")


def main(img_path, model_path, svm):
    img_name = img_path.split("/")[-1]
    model = RCNN()
    state = torch.load(model_path)
    model.load_state_dict(state['model'])
    print(f"Best ACC:{state['best_acc']}")
    img = cv2.imread(img_path)
    rects = selectiveSearch(img)
    print("regions:", len(rects))
    info = [img_name, len(rects)]
    final_bbox, info = generateBbox(img, rects, transforms, model, svm, info)
    img_out = img.copy()
    drawBbox(img_out, final_bbox, info)


if __name__ == "__main__":
    img_path = "./test_img/test3.jpg"
    model_path = "./check_point/best.pth"
    svm_path = "./check_point/svm_cls.pkl"
    with open(svm_path, "rb") as f:
        svm_cls = pickle.load(f)
        f.close()
    main(img_path, model_path, None)
