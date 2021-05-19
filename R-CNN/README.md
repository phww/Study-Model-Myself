# R-CNN 个人实现
[论文连接](https://arxiv.org/abs/1311.2524)
[参考博客](https://towardsdatascience.com/step-by-step-r-cnn-implementation-from-scratch-in-python-e97101ccde55)
[个人博客详解](http://1.116.141.209/index.php/archives/15/)
R-CNN算法的整个流程主要分为4步(原始论文中最后还有一步Bbox回归，这里暂未去实现)：
- **step1：生成候选区域**

  使用Selective Search算法提取图片中的候选区域（Region Proposal）。将其作为train_data

- **step2：生成候选区域数据集**

  每个step1得到的region与原图片标定的bbox（Bounding Box）**计算IOU**。并依据IOU为每个region标定分类label，作为train_label。与step1得到的train_data一起生成候选区域数据集

- **step3：使用CNN网络提取step1和step2得到的候选区域数据集的特征，并用SVM分类器进行区域分类**

  直接Fine Turn VGG16, 作为CNN特征提取网络。使用SVM判断候选区域中是否包含目标物

- **step4：测试**

  先用Selected Search算法提取测试图片的Region Proposal，将前2000个region送入训练好的CNN模型提取特征，并使用SVM分类器预测这些region的分类。从而选出包含目标物的region


