# R-CNN 个人实现
[论文连接](https://arxiv.org/abs/1311.2524)
[参考博客](https://towardsdatascience.com/step-by-step-r-cnn-implementation-from-scratch-in-python-e97101ccde55)
[个人博客详解](http://1.116.141.209/index.php/archives/15/)
R-CNN算法的整个流程主要分为4步(原始论文中最后还有一步Bbox回归，这里暂未去实现)：
- **step1：生成候选区域**

  使用selective search算法提取图片中的候选区域（region proposal）。将其作为train_data

- **step2：计算IOU**

  每个step1得到的region与原图片标定的bbox（Bounding Box）计算IOU。并依据IOU为每个region标定分类label，作为train_label

- **step3：使用CNN网络学习step1和step2得到的train_set**

  直接Fine Turn VGG16, 作为CNN网络。使用训练好的CNN网络提取region中的特征，然后训练SVM分类器

- **step4：测试**

  先用selected search算法提取测试图片的region proposal，将前2000个region送入训练好的CNN模型，获得这些region的分类

