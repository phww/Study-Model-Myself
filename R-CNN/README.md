# R-CNN 个人实现
- R-CNN算法的整个流程主要分为4步(原始论文中最后还有一步Bbox回归，这里暂未去实现)：

  - **step1：生成候选区域和候选区域数据集**
  
    使用Selective Search算法提取图片中的候选区域（Region Proposal）。将其作为train_data，并且将每个得到的region与原图片标定的bbox（Bounding Box）**计算IOU**。并依据IOU为每个region标定分类label，作为train_label。与前面得到的train_data一起生成**候选区域数据集regions**
  
  - **step2：CNN特征提取器**
  
    直接**Fine Turn VGG16,** 作为CNN特征提取网络
  
  - **step3：SVM区域分类器**
  
    用CNN特征提取器提取的特征训练SVM，使用SVM判断候选区域中是否包含目标物
  
  - **step4：测试**
  
    先用Selected Search算法提取测试图片的Region Proposal，将前2000个region送入训练好的CNN模型提取特征，并使用SVM分类器预测这些region的分类。从而选出包含目标物的region
  
  [参考博客](https://towardsdatascience.com/step-by-step-r-cnn-implementation-from-scratch-in-python-e97101ccde55)
  
  [论文（arxiv）](https://arxiv.org/abs/1311.2524)
  
  [github源码](https://github.com/phww/Study-Model-Myself/tree/main/R-CNN)

