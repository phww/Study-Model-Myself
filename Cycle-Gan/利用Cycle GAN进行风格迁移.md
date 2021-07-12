# 利用Cycle GAN进行风格迁移

[本项目]()实践源码[来自](https://github.com/aitorzip/PyTorch-CycleGAN)

## 一、背景

### **什么是风格迁移**

风格迁移指将A类图片的风格转换为B类图片的风格。比如将一张普通的画转换为梵高风格的画，将普通马转换为斑马。这其中涉及到使用已知图片生成另一种风格的图片，即涉及到图片的生成。生成式对抗网络（Generative Adversarial Networks）GAN，正好适合处理这一种任务。

### **普通的GAN网络**

普通的GAN使用生成器（Generator）来将A类型的图片转换为B类型的图片，然后使用鉴别器（Discriminator）来判断生成的B类型的图片和真实的B类型的图片的相似度（得分在0~1）。如下图所示：

![image-20210706100611354](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210706100611354.png)

因此可以假设我们有两个网络，G（Generator）和D（Discriminator）。他们的功能分别是：

- G是一个生成图片的网络，它接收一个随机的噪声z，通过这个噪声生成图片，记做G(z)。
- D是一个判别网络，判别一张图片是不是“真实的”。它的输入参数是x，x代表一张图片，输出D（x）代表x为真实图片的概率，如果为1，就代表100%是真实的图片，而输出为0，就代表不可能是真实的图片。

在训练过程中，**生成网络G的目标就是尽量生成真实的图片去欺骗判别网络D。而D的目标就是尽量把G生成的图片和真实的图片分别开来。**这样，G和D构成了一个动态的“博弈过程”。**在最理想的状态下，**G可以生成足以“以假乱真”的图片G(z)。对于D来说，它难以判定G生成的图片究竟是不是真实的，因此D(G(z)) = 0.5。

由于有两个网络，因此普通的GAN网络的损失也有两个。第一个损失为生成器的loss，对于生成器希望生成的图像越逼真越好。另一个损失为鉴别器的loss，对于鉴别器希望他能给真实的图片的得分接近1，而给生成的假图片的得分接近0。以下是GAN原始论文中提出的loss：

$\underset{G}min\ \underset{D}maxV(D,G)=E_{x~p_{data}(x)}[logD(x)] + E_{z~p_z(z)}[log(1-D(G(z)))]$

论文中提出的训练方法：其中鉴别器使用**梯度上升**的方法训练，而生成器使用**梯度下降**的方法训练。且先训练鉴别器再训练生成器。

![image-20210706173335275](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210706173335275.png)



## 二、**Cycle GAN**

### 网络结构

Cycle Gan可以视为两个普通的GAN网络，第一个网络负责将A转换为B，并负责鉴别B。而第二个网络负责将B转换为A，并负责鉴别A。所以一共有两个生成器和两个鉴别器。其结构如下所示：

![image-20210706175935295](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210706175935295.png)

如下图所示，整个Cycle GAN网络中数据流动的路线有4条。因此在训练Cycle GAN时，每一次迭代实际上是同时更新四个模型的参数。

![img](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/20190103164253170.png)

在实际实现中，生成器和鉴别器均采用CNN网络提取特征。本实践中他们具体的网络结构如下：

![image-20210706190837462](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210706190837462.png)



### 损失函数

Cycle GAN的论文正文中提到了两种损失：adversarial losses（对抗损失），即原始GAN的损失，在Cycle GAN中包括GA-B、DB以及GB-A、DA两个方向的网络的对抗损失。和cycle consistency losses(循环一致性损失)，因为希望GA-B网络生成的fake_B能够经由GB-A网络生成fake_A，其中用L1 loss来表示A和fake_A之间的差距。同理GB-A到GA-B之间的B和fake_B也用L1 loss衡量他们之间的差距。

**adversarial losses（对抗损失）：**

![img](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/2019032717035441.png)

![img](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/20190327171610702.png)

同理另外一个方向的损失为：

![img](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/20190327171653323.png)

**cycle consistency losses(循环一致性损失)：**我们希望x -> G(x) -> F(G(x)) ≈ x，称作forward cycle consistency。同理，y -> F(y) -> G(F(y)) ≈ y, 称作 backward cycle consistency。

![img](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/20190327173424105.png)

**网络的总体loss：**其中可以按需求给这三个loss加权，原论文中仅仅给cycle consistency losses加了权重系数。

![img](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/20190327173542569.png)

在论文的附录与代码中还提出和实现了另外一种损失：**identity loss**，即GA_B在处理B类型的图想时也要保持生成的图片几乎不变，对GB-A处理A类型的图片同理。论文中作者认为加入该loss可以防止风格迁移过度。

![image-20210706200640211](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210706200640211.png)



## 三、实例结果

使用 summer2winter_yosemite数据集，进行冬日到夏日风格图片的转换。其中的一些数据如下：

**夏日：**

<img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/2012-07-31 21:49:20.jpg" width="350" style="float:left"/><img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/2012-06-10 11:56:53.jpg" width="350" style="float:right"/>







<img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/2014-07-01 23:36:11.jpg" width="350" style="float:left"/><img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/2013-05-22 00:17:20.jpg" width="350" style="float:right"/>

























**冬日：**
<img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/2013-08-01 00:20:40.jpg" width="350" style="float:left"/><img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/2013-06-23 11:12:40.jpg" width="350" style="float:right"/>
<img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/2011-07-10 15:39:59.jpg" width="350" style="float:left"/><img src="https://pic-1305686174.cos.ap-nanjing.myqcloud.com/2012-01-24 17:18:38.jpg" width="350" style="float:right"/>































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





























