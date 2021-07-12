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

## 一、模型和数据集

#### Show Attend And Tell 提出的模型

![image-20210628203755806](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210628203755806.png)

如上图所示：本实践主要魔改原模型的第一、二步骤：即输入的数据不在是一张图片，而是从一个视频中提取的k帧图片。这k帧图片被分别送入第二步的CNN网络中提取特征，待所有帧的特征图被提取完毕后，将这些特征图通过取"**均值**"的方式**融合在一起**。之后的操作和原论文的一模一样。

#### 数据集

数据集由课程老师提供。数据集组织方式类似于MSR-VTT数据集。即每段视频为10~30s的短视频，为每段视频事先标注了5个英文的描述。同时也提供了多模态的数据：手语和声音，但是本实践并没有使用多模态的数据。本数据集分为训练集、验证集和测试集。**其中训练集有1900个视频、验证集有300个视频，且训练集和验证集都包含事先标注的caption。而测试集中有800个视频，且没有caption**。并通过参加竞赛，使用测试集生成的caption在竞赛中的得分来评判模型的好坏。

**视频数据：**

![image-20210628150724353](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210628150724353.png)

**事先标注的caption：**

![image-20210628150841335](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210628150841335.png)



## 二、预处理

### 1.提取视频中的关键帧

####  **关键帧的定义**

Show Attends and Tell中的提出的方法，在输入端要求输入的数据格式为一张**图片**，然后为该图片生成一段描述性语句、caption。如果将视频的全部帧输入网络，而每个视频至少有2、3百帧，因此会生成为每个视频生成几百个描述，这显然不合理。而且全部视频帧中有大量类似的场景，对于这些场景只需要其中一帧就足够了。为此需要提取视频的关键帧。所以将视频中的**关键帧定义**为：视频的**第一帧**和视频中**镜头转换**后的第一帧。同时人为设定**每个视频必须提取出k帧，其中不满k帧的视频，使用第0帧填充至k帧**。

#### **如何识别镜头转换？**

**图像的直方图**，可以表示**图像像素的分布**情况。因此可以用图像的直方图来区分不同的图像。图像的直方图可以认为是一维向量（灰度图）或二维的矩阵（RGB三通道）。其形状为(1，256)或（3，256）代表**图像各通道中0~255像素的累计值**

1. 设有一个函数$f(x，y)$，$x，y$代表两个图像的直方图，$f$函数的输出为这两个**图像直方图的相似度**

2. 对于视频中连续的3帧图片A、B、C，$f(A,B)$表示A、B两帧的相似度，$f(B,C)$表示B、C两帧的相似度。则$score = f(B,C)-f(A,B)$代表A、B、C三帧的图像**直方图相似度的差分**
3. 如果一个视频有n帧图片，则可以得到n-2个相似度的差分。以代表从第3帧开始每帧的一个得分
4. 当这个得分高于一定的**阈值**时。就认为该帧为关键帧(从第三帧开始)

图像的直方图和计算两个直方图的相似度，可以使用OpenCv提供的API：**calcHist()和compareHist()**解决。其中compareHist（）有多种计算直方图相似度的方式。但是每个视频的内容都不一样无法设定同一的阈值来筛选关键帧，为此需要一种自动阈值的方法。

#### **自动阈值**

对一个视频的n帧计算得到n-2个得分。统计这n-2个得分的平均值mean和标准差std，使用**mean+std=threshold**作为自动阈值。使用自动阈值筛选关键帧时，如果整割视频**几乎无镜头跳转**(比如一直采访一个人)，那么所有帧的相似度几乎一致，导致自动阈值筛选的关键帧几乎包含所有帧。因此需要规定提取的最大关键帧数量。

#### **效果展示**

![image-20210628153638940](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210628153638940.png)

#### **编程实现（C++）**

源代码：./Video2Imgs/generateKeyFrame.cpp

- 头文件

  ```C++
  #include <iostream>
  #include <opencv2/opencv.hpp>
  #include <algorithm>
  #include <numeric>
  #include <boost/algorithm/string.hpp> //使用split划分字符串
  #include "getFiles.h" // 自己的头文件，用于递归获取文件夹下的所有文件
  using namespace std;
  using namespace cv;
  ```

  

- 计算图像的直方图:输入视频的一帧，输出3个通道的直方图。这里直接输入灰度图，返回灰度图的直方图对本实践的任务而言效果也差不多

  ```C++
  vector<Mat> getHistRGB(Mat& frame) {
    vector<Mat> hist_rgb(3);
    // 分离BGR通道
    Mat channels[3];
    split(frame, channels);
    // 分别计算每个通道的直方图
    const int bins[1] = {256};
    const float inranges[2] = {0, 255};
    const float* ranges = {inranges};
    calcHist(&channels[0], 1, 0, Mat(), hist_rgb[0], 1, bins, &ranges);
    calcHist(&channels[1], 1, 0, Mat(), hist_rgb[1], 1, bins, &ranges);
    calcHist(&channels[2], 1, 0, Mat(), hist_rgb[2], 1, bins, &ranges);
    return hist_rgb;
  }
  ```

- 自动阈值:输入得分，输出得分的均值+标准差

  ```C++
  double autoThreshold(vector<double>& nums) {
    double mean = std::accumulate(nums.begin(), nums.end(), 0) / nums.size();
    double var = 0;
    for (auto num : nums) {
  	var += pow(num - mean, 2);
    }
    var /= nums.size();
    double std = sqrt(var);
    return std + mean;
  }
  ```

- 主函数:用opencv的VideoCapture逐帧读取视频，从第二帧（下标以0开始）判别该帧是否是关键帧，并保存关键帧。关键帧保存名为视频名 + -视频帧.jpg。如 G16000-100.jpg 代表 G16000.mp4 这个视频的第100帧

  ```c++
  int main() {
    int max_frame = 8; // 必须提取k帧
    string dataset_type = "test"; // train or test or val
    string video_root = "/home/ph/Dataset/VideoCaption/";
    vector<string> video_paths;
    getFiles(video_root + dataset_type, video_paths);
    cout << "待处理视频数：" << video_paths.size() << endl;
    for (int i = 0; i < video_paths.size(); i++) {
  	cout << "处理视频：" << i + 1 << " ";
  	// 捕获一个视频
  	string video_path = video_paths[i];
  	vector<string> split_str;
  	boost::split(split_str, video_path, boost::is_any_of("./"));
  	string video_name = split_str[split_str.size() - 2];
  	VideoCapture cap;
  	cap.open(video_path);
  	if (!cap.isOpened()) {
  	  cout << "can not open video!" << endl;
  	}
  
  	// 视频信息
  	int n_frame = cap.get(CAP_PROP_FRAME_COUNT);
  
  	// 三帧直方图差分
  	Mat frame1, frame2, cur_frame;
  	vector<double> all_scores;
  	for (int j = 0; j <= n_frame; j++) {
  	  // 先获取前两帧
  	  if (j == 0) {
  		cap >> frame1;
  		string name = video_root + "generateImgs/" + dataset_type + "/" + video_name + "-" + to_string(j) + ".jpg";
  		imwrite(name, frame1);
  		cout << "保存第" << j << "帧" << " ";
  		continue;
  	  }
  	  if (j == 1) {
  		cap >> frame2;
  		continue;
  	  }
  
  	  // 从第2帧开始计算直方图差分的分数
  	  cap >> cur_frame;
  	  if (cur_frame.empty()) {
  		break;
  	  }
  	  vector<Mat> A, B, C;
  	  A = getHistRGB(frame1);
  	  B = getHistRGB(frame2);
  	  C = getHistRGB(cur_frame);
  
  	  double score1 = 0, score2 = 0;
  	  for (int k = 0; k < 3; k++) {
  		score1 += compareHist(A[k], B[k], HISTCMP_INTERSECT);
  		score2 += compareHist(B[k], C[k], HISTCMP_INTERSECT);
  	  }
  	  all_scores.push_back(abs(score2 - score1) / 3);
  
  	  // 以三帧为窗口，滑动窗口计算整个视频的差分结果
  	  frame2.copyTo(frame1);
  	  cur_frame.copyTo(frame2);
  	}
  
  	// 从视频的第二帧开始，根据全部差分结果和自适应阈值来确定关键帧
  	double threshold = autoThreshold(all_scores);
  	Mat key_frame;
  	int cnt = 1;
  	for (int k = 2; k <= n_frame; k++) {
  	  if (all_scores[k - 1] > threshold) {
  		cap.set(CAP_PROP_POS_FRAMES, k);
  		cap >> key_frame;
  		if (!key_frame.empty()) {
  		  k++; // 一般来说第k帧和第k+1帧差不多，只要一个就行了
  		  cnt++; // 对一个视频最多只要8帧
  		}
  
  		if (key_frame.empty() || cnt > max_frame) {
  		  break;
  		}
  		string name = video_root + "generateImgs/" + dataset_type + "/" + video_name + "-" + to_string(k) + ".jpg";
  		imwrite(name, key_frame);
  		cout << "保存第" << k << "帧" << " ";
  	  }
  	}
  
  	// 凑不够8帧就用第0帧补充
  	while (cnt < max_frame) {
  	  cap.set(CAP_PROP_POS_FRAMES, 0);
  	  cap >> key_frame;
  	  string name = video_root + "generateImgs/" + dataset_type + "/"
  		  + video_name + "-" + to_string(0) + to_string(cnt) + ".jpg";
  	  imwrite(name, key_frame);
  	  cout << "保存第" << 0 << "帧" << " ";
  	  cnt++;
  	}
  	cout << endl;
  	cap.release();
    }
    return 0;
  }
  ```





### 2.为事先给出的视频描述（caption）生成字典（vocab)

”字典“指的是**字符word到唯一标识索引index的一种映射关系**。比如I Love Coding中 I：1，Love：2，Coding：3。这样在计算机中就可以用”1 2 3“这样的字符串表示"I Love Coding"。在python中用两个dict表示“字典”，一个dict为字符到索引的映射**word2idx**， 另一个dict为索引到字符的映射**idx2word**

#### **编程实现**

源代码：./generateVocab.py

**构建字典类Vocabulary**:其中实现 addWord() 函数为字典类添加新的字符

```python
class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def addWord(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
```

**建立字典**

1. 统计全部视频描述中**所有字符出现的次数**，设置一个**阈值**。(只有出现次数大于阈值的字符才被记录到”字典“中。当阈值为0，代表将全部出现过的字符都记录到”字典“中。)
2. 首先将几个**特殊字符**\<pad>,\<start>,\<end>,\<unk>记录到字典中，其中**\<pad>字符一定要第一个记录**，即对应的索引一定要为0。\<start>和\<end>标记语句的开始和结束，\<unk>用于标注字典中不存在的字符。\<pad>用于填充字符串，因为通常都是在字符串后面填0，因此\<pad>对应的索引一定要为0。
3. 然后**将出现次数大于阈值**的字符才记录到”字典“中

注：nltk这个NLP的库有很多实用的工具，下面的代码中就使用了nltk.word_tokenize()将一个字符串分割为一个个字符

```python
def buildVocab(original_text_path="./video/video_demo/demo.json", threshold=0):
    # 用一个hash表来统计原始语料库中每个字符出现的次数
    counter = collections.Counter()
    #
    with open(original_text_path) as f:
        text_dicts = json.load(f)["sentences"]
        for i, text_dict in enumerate(text_dicts):
            tokens = nltk.word_tokenize(str(text_dict["caption"]).lower())
            counter.update(tokens)
            print("\r", f"已tokenize:[{i + 1}]/[{len(text_dicts)}]", end=" ")
        print("\n")
        # 只有语料库中出现次数超过阈值的字符才被记录到字典中
        words = [word for word, cnt in counter.items() if cnt > threshold]
        # 几个基础的特殊符号
        vocab = Vocabulary()
        # 一定要第一个加<pad>保证<pad>对应的索引为0.其他字符随意
        vocab.addWord("<pad>")
        vocab.addWord("<start>")
        vocab.addWord("<end>")
        vocab.addWord("<unk>")
        # 根据words建立vocabulary
        for i, word in enumerate(words):
            vocab.addWord(word)
            print("\r", f"建立Vocab:[{i + 1}]/[{len(words)}]", end=' ')
        return vocab															
```



## 三、定制Dataset

#### **要求**

1. 已知将视频中的关键帧保存在video/generateImgs文件夹内，视频的描述性语句caption保存在video/caption.json文件中，第二节生成的字典也保存在video/vocab.pkl文件中。

2. 每个视频对应有5句caption，在**训练**和**评估**模型时需要**随机选择其中一句**作为模型生成的caption的ground truth。

3. 每个caption的长度不一，需要为每个batch_size的captions做填充：以该批次captions中最长的长度为基础，填充所有caption。为此需要在继承Pytorch的Dataset时重写collect_fn()

   比如有两句句子：“I Love Coding”，“I Study Artificial Intelligence In SEU”。因此需要将第一个句子填充为"I Love Coding \<pad> \<pad> \<pad>"，以下是重写collect_fn()后得到的效果：
   
   ![collect_fn](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/collect_fn.png)

#### **将多帧图像数据融合为一个数据**

**原始的模型只能做单张图片的image caption**

这里和原始Show Attends And tell 中的模型的操作不一样。之前提到过，每个视频会提取k个关键帧。如果使用原始的模型，将会为每帧生成一句描述。而最后只能从这k句描述中选择一个作为视频生成的caption。**据实验，选择“第0帧生成的描述“”比选择“最长的描述“、“随机选择”最后的竞赛得分最高**。猜测原因是：人工为视频标定caption时，视频开头的片段先入为主占据了人们主要描述的内容。这也是为什么前面，要用第0帧填充不够的关键帧。

**魔改原始模型：融合多帧图片**

为了融合多帧图片，在定制Dataset这个阶段需要：**将每个视频的k帧关键帧在通道维度上堆叠起来，成为一个数据。**即最后每个视频融合后的数据形状为(3*k, 帧大小，帧大小)

#### **编程实现**

源代码: ./dataset.py

```python
class MyCaptionDatasetRaw(Dataset):
    def __init__(self, vocab_path, image_root, caption_path, transforms):
        super(MyCaptionDatasetRaw, self).__init__()
        # 读取各种文件
        self.vocab = Vocabulary()
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.img_root = image_root
        self.img_paths = os.listdir(image_root)
        with open(caption_path, "rb") as f:
            self.text_dicts = json.load(f)["sentences"]
        self.transforms = transforms

    def __getitem__(self, idx):
        # 按下标索引图片
        img_name = self.img_paths[idx]
        img = Image.open(os.path.join(self.img_root, img_name)).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        # 按下标索引图片的描述/caption，ground truth
        caption = []
        # 每个图片有5个caption，随机选一个
        sen_id = random.choice(range(5))
        for i, text_dict in enumerate(self.text_dicts):
            # 按图片名暴力搜索对应的caption
            if text_dict["video_id"] in img_name and text_dict["sen_id"] == sen_id:
                caption = text_dict["caption"]
                break
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())

        target = []
        target.append(self.vocab.word2idx["<start>"])
        # 小心append会将caption正文当做一个列表加入到target中！这里用extend更符合要求
        target.extend(self.vocab.word2idx[word] for word in tokens)
        # 当generateVocab.py里面设置词频阈值大于0时。语料库中的一些词不会被记录到字典中。因此设置其为<unk>
        for word in tokens:
            if word not in self.vocab.word2idx.keys():
                word = "<unk>"
            target.append(self.vocab.word2idx[word])
        target.append(self.vocab.word2idx["<end>"])
        target = torch.tensor(target)
        return img, target

    def __len__(self):
        return len(self.img_paths)


class MyCaptionDatasetFusion(Dataset):
    def __init__(self, vocab_path, image_root, caption_path, transforms, k=8):
        super(MyCaptionDatasetFusion, self).__init__()
        self.k = k
        # 读取各种文件
        self.vocab = Vocabulary()
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.img_root = image_root
        self.img_paths = os.listdir(image_root)
        self.img_paths = sorted(self.img_paths)  # 先排序图片路径
        with open(caption_path, "rb") as f:
            self.text_dicts = json.load(f)["sentences"]
        self.transforms = transforms

    def __getitem__(self, idx):
        # 按下标索引视频提取的所有帧，因为提前排过序。[idx*k，（idx+1）*k]的图片都属于一个视频中提取的关键帧
        video_id = self.img_paths[idx * self.k].split("-")[0]  # G_16000-1.jpg -> G_16000
        img = Image.open(os.path.join(self.img_root, self.img_paths[idx * self.k])).convert('RGB')
        if self.transforms is not None:
            imgs = self.transforms(img)  # 3, 224, 224
        for i in range(1, self.k):
            img_name = self.img_paths[idx * self.k + i]
            img = Image.open(os.path.join(self.img_root, img_name)).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
                imgs = torch.cat([imgs, img], dim=0)  # 3*k, 224, 224

        # 按下标索引图片的描述/caption，ground truth
        caption = []
        # 每个视频有5个caption，随机选一个
        sen_id = random.choice(range(5))
        for i, text_dict in enumerate(self.text_dicts):
            # 按图片名暴力搜索对应的caption。（这里可以二分搜索优化一下）
            if text_dict["video_id"] == video_id and text_dict["sen_id"] == sen_id:
                caption = text_dict["caption"]
                break
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())

        target = []
        target.append(self.vocab.word2idx["<start>"])
        # 当generateVocab.py里面设置词频阈值大于0时。语料库中的一些词不会被记录到字典中。因此设置其为<unk>
        for word in tokens:
            if word not in self.vocab.word2idx.keys():
                word = "<unk>"
            target.append(self.vocab.word2idx[word])
        target.append(self.vocab.word2idx["<end>"])
        target = torch.tensor(target)
        return imgs, target

    def __len__(self):
        return len(self.img_paths) // self.k


def collect_fn(data):
    """
    因为caption的长短不一，而Dataset要求数据的形状是一样的
    为此需要为Dataset重写一个堆叠函数collect_fn
    Args:
        data: list
            Dataset按下标索引返回的一个batch_size的对象，即长度为batch_size的(img, target)列表
    Returns:
        按照一个批次的caption的长度排序后的数据：
        imgs: shape"B 3*k 224 224"
        targets: 经过Vocab编码和填充过后的caption shape"B max_length"
        lengths: caption的原始长度。包含<start>和<end>
    """
    # 先按target的长度从大到小排序data
    data.sort(key=lambda x: len(x[1]), reverse=True)

    # 图片的堆叠方式不变
    imgs, captions = zip(*data)
    imgs = torch.stack(imgs, dim=0)

    # caption以最长的语句为标准，因为定义了"<pad>"字符的idx为0。在不够长度的在句子后面填0，
    lengths = [len(caption) for caption in captions]

    # 用Pytorch提供的API填充语句
    targets = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)

    # 自己写也很容易
    # max_length = max(lengths)
    # targets = torch.zeros(len(captions), max_length).long()
    # for i, caption in enumerate(captions):
    #     cur_len = lengths[i]
    #     targets[i, :cur_len] = caption[:cur_len]

    return imgs, targets, lengths


def get_loader(vocab_path, image_root, caption_path, batch_size, transforms, use_fusion=True):
    """
    返回数据集的dataloader
    Args:
        vocab_path: 预处理得到的字典文件path.格式为pkl
        image_root: 预处理视频得到的关键帧图片的root
        caption_path: 原始的caption文件。格式为json

    Returns:
        dataloader:dataset的迭代器，每次迭代返回一个批次的imgs, targets, lengths
    """
    if use_fusion:
        data_set = MyCaptionDatasetFusion(vocab_path, image_root, caption_path, transforms=transforms)
    else:
        data_set = MyCaptionDatasetRaw(vocab_path, image_root, caption_path, transforms=transforms)
    data_loader = DataLoader(data_set, batch_size, shuffle=True, pin_memory=True, collate_fn=collect_fn, num_workers=6)
    return data_loader
```





## 四、搭建网络

**整儿网络的预览**

![image-20210704150829580](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210704150829580.png)

### 1.CNN模块

#### **目的**

CNN模块作为特征提取器(extractor)，接受**融合了的视频数据**作为输入。输出L个与融合数据的一部分有关向量，每个向量的维度为D。用$a$表示这个输出，则：$a=\{a_1,...,a_L\}, a_i\in R^D$。可以理解$a_i$表示融合数据第$i$部分的特征向量。

**过程主要分为两步：**

- 直接fine turn restnet101，取restnet101的最后的卷积层的输出的(B, 2048, 2, 2)的特征图作为提取的特征（编码的特征）。

- 将融合的数据拆开，依次送入CNN中提取特征图。但是不直接将特征图送入下一个模块，而是等待CNN提取融合数据中的全部特征图，并将其在第二个维度上堆叠起来。即最终形状为(B,k,2048,2,2)。最后在第二个维度上**取平均值**，将最终输出作为每个视频经过CNN后被提取的融合特征。

- 注：

  1.得到形状为(B,k,2048,2,2)的融合特征图后，还可以使用3维卷积的方法来融合所有特征。本文没有实践这个方法。

  2.经实验使用原论文中使用的VGG作为CNN特征提取网络，效果不如resnet101.

  3.经实验由于数据集不够大，重新训练CNN网络的最后几层的效果不如直接使用Pytorch预训练的完整的网络。

```python
class CNNEncoder(nn.Module):
    """CNN提取图片的特征，使用fine turn的resnet101"""

    def __init__(self, encode_image_size=14, fine_turn_layer=-3, cnn_type="resnet"):
        super(CNNEncoder, self).__init__()
        if cnn_type == "resnet":
            resnet = resnet101(pretrained=True)
            # 不需要resnet101的最后两层
            modules = list(resnet.children())[:-2]
            # self._fine_turn(fine_turn_layer) # 当前使用的训练集太小，重新训练最后几层反而效果不好
            self.n_channels = 2048

        # 实验VGG效果不如resnet101
        elif cnn_type == "vgg":
            vgg = vgg16_bn(pretrained=True)
            modules = list(vgg.children())[:-2]
            self.n_channels = 512
        self.extract_feat = nn.Sequential(*modules)
        # resnet倒数第二层的输出形状为（b，2048， 2， 2）需要变为（b，2048， 14， 14）
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(encode_image_size, encode_image_size))

    def forward(self, images, mean_fusion=True):
        """
        Args:
            images: shape"B, 3xk, 224, 224", 3xk代表一个视频提取了k帧图片，k=1时代表不融合多帧图片
            mean_fusion: 融合多帧图片，融合后，shape"B, 3, 224, 224"
        Returns:
            output: CNN提取的feature map. shape"B, 14, 14, 2048"

        """
        # 用CNN提取特征k帧的特征
        k = images.size(1) // 3
        features = self.extract_feat(images[:, :3, :, :]).unsqueeze(dim=1)  # B, 1, 2048, 2, 2
        for i in range(1, k):
            feature = self.extract_feat(images[:, i * 3:(i + 1) * 3, :, :]).unsqueeze(dim=1)
            # 在dim=2堆叠一个视频中的k帧的feature map
            features = torch.cat([features, feature], dim=1)  # B, k, 2048, 2, 2
        # 取平均，融合特征
        if mean_fusion:
            features = features.mean(dim=1).squeeze(dim=1)  # B, 2048, 2, 2

        # 调整形状为(B, 14, 14, 2048)
        features_2048x14x14 = self.pooling(features)
        output = rearrange(features_2048x14x14, "B N H W -> B H W N")
        return output

    # fine turn最后3层，即特征向量长度为2048的卷积层
    def _fine_turn(self, fin_turn_layer):
        for params in self.extract_feat.parameters():
            params.requires_grad = False
        for params in list(self.extract_feat.modules())[fin_turn_layer:]:
            params.requires_grad = True
```



### 2.Attention模块

#### **目的**

在原论文中attention机制分为hard attention和soft attention。本实践只使用soft attention。attention 模块可以为CNN模块输出的每个分量$a=\{a_1,...,a_L\}, a_i\in R^D$生成对应的权重，**代表第i个位置对生成下一个词的重要性**。设权重为$a_i$,则attention模块最终输出**语义向量(context vector)**$\hat{z_t}=\sum_{i=1}^L{{a_i}*{\alpha_i}}$。即将语义向量就是将CNN提取的特征加权。

#### **Attention机制**

那么如何加权？设CNN模块的输出为：$a=\{a_1,...,a_L\}, a_i\in R^D$，而RNN网络每个时间步的隐藏态为$h=\{h_1,...h_t\}$。为此可以定义一个**MLP**网络$f_{att}$来处理这两种数据，让他们产生联系。因此令:$e_{ti}=f_{att}(a_i,h_{t-1})$，之后归一化$e_{ti}$，最终定义attension为$\alpha_{ti}=\frac{exp(e_{ti})}{\sum_{k=1}^L exp(e_{tk})}$。

#### **代码实现**

```python
class Attention(nn.Module):
    """注意力模块：通过CNN提取的feature map和RNN每个隐藏态的关系得到attention权重，为feature map的每个像素加权"""

    def __init__(self, att_dim, encoder_out_dim, decoder_dim):
        super(Attention, self).__init__()
        # 统一CNN输出、rnn隐藏态的特征向量的长度为att_dim
        self.encoder_att = nn.Linear(encoder_out_dim, att_dim)
        self.hidden_att = nn.Linear(decoder_dim, att_dim)
        # 将attention的特征维度降为1
        self.full_att = nn.Linear(att_dim, 1)
        # relu激活后使用softmax将attention权重归一化，这样权重之和为1。才符合权重的概念
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, rnn_hidden_state):
        """
        Args:
            encoder_out: CNN编码器提取的图像特征,shape"B, 14x14, 2048".
                这里将H和W维度压缩到了一起。可以理解为：特征图有14*14个像素，每个像素的特征向量的维度为2048
            rnn_hidden_state: RNN解码器的隐藏态，shape"B, 2048"

        Returns:
            encoder_out_weighted: 加了权的图像特征，shape"B, 2048"
            att_weight：权重矩阵，用于可视化网络进行推断时更关注原始图像的哪个部分（像素）,shape“B, 14*14”

        """
        encoder_att = self.encoder_att(encoder_out)
        # 下面要求和先扩充对应的维度
        rnn_hidden_att = self.hidden_att(rnn_hidden_state).unsqueeze(dim=1)
        # 经过full_att后输出形状为"B, 14x14, 1"，最后的1我们将其压缩掉
        att = self.full_att(self.relu(encoder_att + rnn_hidden_att)).squeeze(dim=2)
        # 归一化的attention才能称为权重，其shape"B, 14*14"
        att_weight = self.softmax(att)
        # 加权CNN提取的特征,因为encoder_out.shape=(B, 14*14, 2048),而权重shape=(B, 14*14)
        # 因此权重要先在最后加一个维度，最终加权输出形状为(B, 2048)
        encoder_out_weighted = (encoder_out * att_weight.unsqueeze(dim=2)).sum(dim=1)
        return encoder_out_weighted, att_weight
```



### 3.LSTM解码模块

在训练阶段采用导师驱动（teacher force）的方式训练模型。LSTM网络**初始的$h_0$和$c_0$，**由CNN模块输出的特征图确定。每个时间步，将**加权过后的融合特征和ground truth在此时刻对应的词向量**送入LSTM cell。具体实现见下图：

![image-20210704150719960](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210704150719960.png)

#### 代码实现

```python
class RNNDecoderWithAttention(nn.Module):
    def __init__(self, att_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(RNNDecoderWithAttention, self).__init__()
        self.att_dim = att_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size
        # attention模块
        self.attention = Attention(att_dim, encoder_out_dim=encoder_dim, decoder_dim=decoder_dim)
        # 词嵌入模块
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        # lstm模块
        self.lstm_pre_step = nn.LSTMCell(input_size=encoder_dim + embed_dim, hidden_size=decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # 用于构建gate gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # 用于将隐藏态h转换为当前时刻预测的词向量
        # 初始化词嵌入层和全连接层的参数
        self._init_weights()

    def _init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        RNN初始状态的h0与c0定义为CNN模块输出的feature map
        在第二个维度上的平均值，并且用线性映射将特征向量的维
        度映射到decoder_dim的维度
        """
        mean = encoder_out.mean(dim=1)
        h0 = self.init_h(mean)
        c0 = self.init_c(mean)
        return h0, c0

    def forward(self, encoder_out, captions_gt, captions_length):
        """
        Args:
            encoder_out: CNN提取的特征图. shape"B, 14, 14, 2048"
            captions_gt: 预处理过后的caption的ground truth. shape"B, max(captions_length), vocab_size"
                在数据读取阶段对caption的处理：对一批原始的captions，先用vocab将它们映射为对应字符的idx，然后按
                它们的长度从大到小排序，并在每个caption的头尾加上特殊的token"<start>"和"<end>",最后用"<pad>"
                填充其他长度不够的caption。即"i love coding"->"<start> "i love coding <end><pad>..."
                -> "1 2 3 4 0 0 0..."
            captions_length: shape“B，”
                一个批次的captions，未被填充前的长度。包括<start>和<end>两个token。

        Returns:
            pred_word_vec: 每个批次网络预测的词向量 shape"B, max(decode_length), vocab_size"
            caption_gt: shape"B, max(decode_length)+1"
            decode_length: shape"B,"
            att_weights: shape"B, 196"

        """
        batch_size = encoder_out.size(0)
        # 词嵌入
        caption_gt = captions_gt.to(dtype=torch.long)
        caption_embed = self.embedding(caption_gt)  # "B, max(captions_length), embed_dim"
        # 展开特征图shape “B, 14, 14, 2048”->“B, 14*14, 2048”
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)
        pixels = encoder_out.size(1)  # 196
        # 初始的h0和c0
        h, c = self.init_hidden_state(encoder_out)  # "B, encoder_dim"

        # 原始句子长度需要减去"<start>"token
        decode_length = [c - 1 for c in captions_length]
        max_length = max(decode_length)
        # LSTM
        pred_word_vec = torch.zeros(batch_size, max_length, self.vocab_size)
        att_weights = torch.zeros(batch_size, max_length, pixels)

        for t in range(max_length):
            # 每个时刻只计算原始句子长度比当前时刻小的句子
            batch_size_t = sum([l > t for l in decode_length])
            # 使用t-1时刻的隐藏态h与CNN提取的feature map计算attention。并使用这个att加权feature map
            att_weighted_encoder_out, att_weight = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            att_weighted_encoder_out *= gate
            # 更新t时刻的h和"记忆细胞"c
            # lstm每一时刻的输入为：加了t时刻attention权重的CNN提取的feature map（B_t，2048）
            # 和该时刻应该输入的caption(B_t,decoder_dim).两者堆叠出来的特征(B_t, 2048 + decoder_dim)
            h, c = self.lstm_pre_step(torch.cat([caption_embed[:batch_size_t, t, :], att_weighted_encoder_out], dim=1)
                                      , (h[:batch_size_t], c[:batch_size_t]))
            # 用t时刻的h预测
            preds = self.fc(self.dropout(h))  # B, vocab_size
            # 保存每个时刻的结果
            pred_word_vec[:batch_size_t, t, :] = preds
            att_weights[:batch_size_t, t, :] = att_weight
        return pred_word_vec, caption_gt, decode_length, att_weights
```

## 五、训练和推断

训练代码见./train.py。推断代码见：./inference.py

#### 1.**模型与调参**

使用如下的参数训练50epoch后的模型最终取得了最好的效果

```python
batch_size_train:5
batch_size_val:5
encoder_init_lr:0.0001
decoder_init_lr:0.0001
att_dim:512
decoder_dim:512
embed_dim:512
use_fusion:True
k=8
```

- 首先是完全使用Show Attend And Tell 中的模型，即每个视频提取8帧图片，然后为这8帧都生成一句描述。最后从这8句描述中选择一句作为整个视频的描述。其中选择方式有：选第一帧的描述，选最长的描述和随机选择一个描述。经实验选择第一帧图生成的描述作整个视频的描述效果最好，推测原因是人为标定数据集时，将大多数描述都集中在视频的开始片段。以这种方法训练的模型在竞赛中经过调参能取得0.207左右的成绩。
- 使用mean fusion的方式将8帧图片经过CNN后生成的特征融合。然后使用这一融合特征生成一句描述作为整个视频的描述。该方法经过调参后能够取得0.227左右的成绩。但是该方法的数据单位为视频不是帧，因此只有1900个训练集。导致该方法非常容易过拟合且在训练集和测试集上的损失曲线都过早的收敛。

#### 2.推断

![image-20210704151720412](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210704151720412.png)

在推断时，LSTM模块不使用训练阶段的"导师驱动"模式而是使用“free run mode”。因此不能直接使用事先定义的LSTM模块进行推断，需要将LSTM模块中的各种子模块单独取出从新组合，以进行推断。同时，使用“free run mode”进行推断，得到的答案是不唯一的。为此在推断阶段，很容易想到两种搜索"答案"的方法：**贪心和穷举**。

**贪心和穷举搜索**：设t时刻推断出来的词向量的分布情况为$p(y_t|X,y_1,y2,...y_{t-1})$，一般这个数值的形状为（1，字典大小），比如字典中就3个词i，love，coding。对应的概率分布为（0.5,0.3, 0.2）。则贪心搜索会选择概率最大的词作为t时刻推断出来的$y_t$。不断推进时间步，最终得到的文字序列中的每个位置上的词都是该时刻概率最大的词。穷举顾名思义，每个时间步的所有概率的词都要考虑，这其实是一个**广度优先搜索（BFS）**的过程。**穷举时间复杂度太高**首先淘汰。而**贪心搜索**可以认为是每一个词都取最优，而对于最后推断出来的句子，这种**局部最优不一定是全局最优**（暂时没有证明，但是我的实验结果确实是符合这一现象的）。为此需要使用**beam search(束搜索)**算法来高效的搜索出高质量的描述。[参考文档](https://github.com/phww/Study-Model-Myself/blob/main/Show-Attend-And-Tell/markdown/NLP%E5%B8%B8%E7%94%A8%E7%AE%97%E6%B3%95Beam%20Search%EF%BC%88%E6%9D%9F%E6%90%9C%E7%B4%A2%EF%BC%89.md)



## 六、结果与总结



#### 1.竞赛结果

![image-20210704153901009](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210704153901009.png)

我们的模型最终取得了第五（共35个队伍）的成绩，在几个评分指标中。由于我们的模型只考虑了BLEU4，所以BLEU4指标比较高。而其他指标均没有考虑，导致其他指标得分几乎垫底。

#### **2.最优模型结果**

- 每个epoch训练集和验证集上的平均loss：红线为验证集，蓝线为训练集。

  ![avg_epoch_loss (1)](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/avg_epoch_loss (1).svg)

  可以看出模型在20、30epoch后就已经过拟合了。且在训练集和测试上均在40epoch左右收敛

- 验证集上的BLEU4指标

  ![eval_bleu](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/eval_bleu.svg)

  最大bleu4也就0.06左右，但是此时的模型已经严重过拟合。在800个测试集上只会重复从5、6个句子推断一个作为最终结果。最终在竞赛中能取得0.16左右的成绩。部分结果如下：

  ![image-20210704153228059](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210704153228059.png)

  最后使用epoch30的模型继续训练调参，最终得到bleu4在0.056左右的模型。此模型推断出来的描述就要多样性一些，且最后在竞赛中取得了0.227的成绩。部分推断结果如下：

  ![image-20210704152920613](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210704152920613.png)

![image-20210704152934737](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210704152934737.png)

#### 3.**总结**

**不足**

- 本实践虽然魔改了Show Attend and Tell这篇论文中的模型，但是使用融合的特征导致数据集太小。因此训练模型时非常容易过拟合。而且当时时间和计算设备有限，也没有使用样本足够丰富的MSR-VTT数据集预训练模型，然后在使用自己的模型fine turn。
- 在融合特征时，仅仅简单的使用取“平均”的方法融合特征。没有使用到帧与帧之间的时序信息。据了解，高分模型在提取特征时使用了“3维卷积来处理一个视频中多帧的图片”。

**收获**

- 学习到了很多NLP领域常用的方法：定义词典，词嵌入，基于RNN和门控RNN的语言模型，teacher forcing 和beam search算法。对NLP的模型开发有了更直观的认识。

