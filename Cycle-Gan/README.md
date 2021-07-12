# 使用Cycle GAN进行夏日和冬日的风格转换

[本项目](https://github.com/phww/Study-Model-Myself/tree/main/Cycle-Gan)实践源码来自[这个项目](https://github.com/aitorzip/PyTorch-CycleGAN)

### 训练结果

#### LOSS曲线

**生成器的总loss**：即将循环一致性损失对抗损失和identity loss加权结合起来的总loss

![loss_G](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/loss_G.svg)

**鉴别器的总loss**

![loss_D](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/loss_D.svg)

**循环一致性损失**

![loss_G_cycle](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/loss_G_cycle.svg)

**对抗损失**：对抗损失就是原始GAN网络的生成器的loss，根据对抗损失和鉴别器的损失曲线可以看出。这个两个loss是相互制约的，一般网络训练到一定程度时。对抗损失会增加，鉴别器的损失会下降。

![loss_G_GAN](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/loss_G_GAN.svg)

**Identity loss**

![loss_G_identity](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/loss_G_identity.svg)

#### **生成的图片效果：**左边为真实图片，右边为风格转换过后的图片

**夏日->冬日：**

<img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/2011-05-28 15:13:21.jpg" width="350" style="float:left"/><img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/0006.png" width="350" style="float:right"/>











<img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/2011-08-18 18:37:21.jpg" width="350" style="float:left"/><img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/0028.png" width="350" style="float:right"/>





<img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/2014-07-05 20:12:01.jpg" width="350" style="float:left"/><img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/0163.png" width="350" style="float:right"/>































**冬日->夏日**

<img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/2013-12-05 00:11:10.jpg" width="350" style="float:left"/><img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/0129.png" width="350" style="float:right"/>

<img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/2015-11-27 19:32:30.jpg" width="350" style="float:left"/><img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/0162.png" width="350" style="float:right"/>

<img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/2016-02-04 14:41:51.jpg" width="350" style="float:left"/><img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/0195.png" width="350" style="float:right"/>













































#### 不足

- 原始图片和生成的图片分辨率都是256x256，但是生成的图片明显没有原始图片细腻。有比较明显的粗糙感。有论文指出通过堆叠多层的鉴别器可以提高生成图片的质量。但是本实践只有一个鉴别器（DA和DB）
- 风格迁移的过程中无法指定迁移对象，导致生成一些奇怪的图片。比如下图中“长苔藓的狼”

<img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/2012-02-29 22:20:01.jpg" width="350" style="float:left"/><img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/0073.png" width="350" style="float:right"/>

<img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/2013-01-02 19:16:00.jpg" width="350" style="float:left"/><img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/0097.png" width="350" style="float:right"/>



















