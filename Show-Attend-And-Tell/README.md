# 使用Show Attends and Tell 实现 Video/Image Caption

>
>
>Video/Image Caption指的是为视频或图片自动生成一段描述性语句。这是一个CV和NLP都有涉及的任务。这里使用[Show Attends and Tell]() 这篇论文中提出的方法和模型：使用CNN网络提取图片的特征->Attention模块：每个时间步用attention加权CNN提取的特征->使用LSTM网络生成一段描述性语句。并基于Video Caption任务对模型进行了一定的魔改
>
>其中attention由CNN提取的特征图和LSTM每个时间步中的隐藏态进行计算得到，其形状和特征图一样。因此直接和原特征图做点乘即可加权特征图。
>
>本文主要分为一下几个部分:
>
>1. 预处理：提取视频中的关键帧和为事先标注的caption生成对应的字典（Vocab）
>2. 自定义数据集
>3. 搭建模型
>4. 训练和推断
>5. 结果分析
>
>[github](https://github.com/phww/Study-Model-Myself/tree/main/Show-Attend-And-Tell)

### Show Attend And Tell 提出的模型

![image-20210628203755806](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210628203755806.png)

如上图所示：本实践主要魔改原模型的第一、二步骤：即输入的数据不在是一张图片，而是从一个视频中提取的k帧图片。这k帧图片被分别送入第二步的CNN网络中提取特征，待所有帧的特征图被提取完毕后，将这些特征图通过取"**均值**"的方式**融合在一起**。之后的操作和原论文的一模一样。

### **魔改后的模型**

![image-20210704150829580](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210704150829580.png)

将每个视频的k帧关键帧依次送入CNN中提取特征图。但是不直接将特征图送入下一个模块，而是等待CNN提取融合数据中的全部特征图，并将其在第二个维度上堆叠起来。即最终形状为(B,k,2048,2,2)。最后在第二个维度上**取平均值**，将最终输出作为每个视频经过CNN后被提取的融合特征。

