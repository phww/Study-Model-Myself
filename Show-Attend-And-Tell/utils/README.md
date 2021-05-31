# 个人使用的Pytorch训练模板

[魔改自这个github](https://github.com/KinglittleQ/Pytorch-Template)

[我的github](https://github.com/phww/tutorials-and-utils/tree/main/utils/Pytorch-training-template)

## 使用方法

1. 下载Pytorch-training-template文件夹下的文件，推荐将文件夹改名为utils。然后将文件夹复制到模型工程目录下

2. 在训练模型代码中继承模板类并重载\__init__()函数，如下：

   ```python
   from utils.template import TemplateModel 
   from torch.utils.tensorboard import SummaryWriter
   # 数据集
   train_loader = ...
   test_loader = ...
   # 模型信息
   model = ...
   optimizer= ...
   loss_fn = ...
   class Trainer(TemplateModel):
       def __init__(self):
           super(Trainer, self).__init__()
           # tensorboard
           self.writer = SummaryWriter()
           # 训练状态
           self.global_step = 0
           self.epoch = 0
           self.best_acc = 0.0
           # 模型架构
           self.model = model
           self.optimizer = optimizer
           self.criterion = loss_fn
           # 数据集
           self.train_loader = train_loader
           self.test_loader = test_loader
           # 运行设备
           self.device = "cuda" if torch.cuda.is_available() else "cpu"
           # check_point 目录
           self.ckpt_dir = "./check_point"
           # 训练时print的间隔
           self.log_per_step = 100
   ```

3. 生成Trainer的一个实例trainer，主要使用train_loop()和eval()两个成员函数，同时注意是否要继续训练某个模型。如下：

   ```python
   epochs = 10
   def train(continue_training=False, continue_model=None):
       trainer = Trainer()
       trainer.check_init()
       trainer.get_model_info(fake_inp=torch.randn(1, 1, 32, 32))
       # 继续训练某个模型
       if continue_training:
           trainer.load_state(continue_model)
       for epoch in range(epochs):
           print(f"Epoch {epoch + 1}\n-------------------------------")
           trainer.train_loop()
           trainer.eval(save_per_epochs=5)
   ```

4.训练完成后模型工程根目录下会有runs文件夹，里面是Tensorboard的文件。在模型工程根目录下的命令行中输入**tensorboard --logdir runs**查看模型训练曲线等信息，如下所示：

![image-20210527213159772](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210527213159772.png)

![image-20210527214955077](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210527214955077.png)

![image-20210527215024350](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210527215024350.png)



## 注意事项

模板代码逻辑见代码注释，这里说几个容易出bug的点

### 单独取出模型

保存的.pth文件里面不只是有模型参数，因此如果要**单独用训练好的模型进行推断**，有两个方法。如下:

``` python
# 1.用直接读取后，用模板中的inference(self, x)方法
model_T = torch.load("best.pth")
pred = model_T.inference(inp)
# 2.读取模型参数后正常使用
model = ...
state = torch.load("best.pth")
model.load_state_dict(state['model'])
pred = model(inp)
```



### 继续训练

训练过程中每次使用eval()评估当前epoch下的模型时，会将准确率最高的模型保存在check_point目录下的best.pth文件中。同时每隔几个epoch（比如1就代表每次评估模型时都保存当前的模型状态）调用eval()时，会保存当前模型的状态在check_point目录下的epoch+数字.pth文件中。

best.pth文件不会保存当前模型在优化器中的参数（因为很占空间），只有epoch+数字.pth文件才会保存优化器中的参数。所以**要继续训练务必使用epoch+数字.pth文件**。

如图保存了优化器参数的模型文件会很大：

![image-20210527214640805](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210527214640805.png)





### 计算训练集中一个批次的loss

train_loss_per_batch(self, batch)函数负责计算训练集中一个批次的loss，这个部分是可能要按需求修改的部分。因为Pytorch 中的loss函数中一般规定x和y都是float，而有些loss函数规定y要为long（比如经常用到的CrossEntropyLoss）如果[官网](https://pytorch.org/docs/stable/nn.html#loss-functions) 对y的数据类型有要求请做相应的修改。本模板除了CrossEntropyLoss将y的数据类型设为long外其他都默认x和y的数据类型为float



### 计算测试集的性能指标metric

本模板使用eval_scores_per_batch(self, batch)配合metric(self, pred, y)函数计算测试集中一批数据的metric，因为不同任务的性能指标太难统一了，这里只是实现了**多分类任务求准确率**的方法。其他任务请按需求继承这个类的时候再重载metric()函数，注意metric()函数返回数据类型为字典,且一定要有准确率acc这个指标，因为acc用于保存训练过程中的最优模型。这个模板使用**分批计算**metric再求全部批次的平均值的策略得到整体的metric。不会将全部的预测和ground truth保存在preds和ys中然后在cpu上进行预测。因为如果测试集或验证集太大（>50000）可能CPU内存装不下，训练会报错.**但是有的metric可能不能使用分批得到的metric求平均来表示整体的metric**,按需求改吧



### 模型summary的BUG

get_model_info()函数内使用了torchsummary这个包中的summary()方法。貌似含有Transformer结构的模型该方法会报错。但是不会影响writer.add_graph()方法在Tensorboard中绘制带Transformer结构的模型的计算图。

![image-20210527215825803](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210527215825803.png)

