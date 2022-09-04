# 各种分类模型学习笔记

## MNASNet

提出移动网络的神经网络架构搜索方法, 该方法的特点如下

1. 将设计问题转为多目标优化问题, 同时考虑准确率和实际推理耗时. 
2. 之前的搜索方法大都先收缩最优的单元, 然后堆叠成网络, 虽然这样能够搜索空间, 但抑制了层多样性. 为解决这个问题, 论文提出分解的层次搜索空间(factorized hierarchical search space), 使得层能存在结构差异的同时, 任然能很好地平衡灵活性和搜索空间.

### Mobile Neural Architecture search

![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\Factorized Hierarchical search space.jpg)

论文提出分别的层次搜索空间, 整体结构如图4所示, 将卷积网络模型分解成独立的块, 逐步降低块的输入以及增加块中的卷积核数. 每个块进行独立块搜索, 每个块包含多个相同的层,由块搜索来决定. 

搜索的目的是基于输入和输出的大小, 选择最合适的算子以及参数(kernal size, filter size)来达到更好的accurate-latency trade-off

![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\MNASNet-architecture search.jpg)

每个块的子搜索包含上面6个步骤, 例如上上图中的block 4, 每层都为inverted bottleneck 5x5 convolution和residual skip path, 共四层.

![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\MNASnet-struct-1.jpg)

搜索空间选择使用mobileNetV2作为参考, 图中的block数于MobileNetV2对应, MobileNetV2的结构上. 再MobileNetV2的基础上, 每个block的layer数量进行$\{0,+1,-1\}$​ 进行加减, 而卷积核数则选择$\{0.75,1.0,1.25\}$

论文提出的分解的层次搜索空间对于平衡层多样性和搜索空间大有特别的好处, 

### Result

![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\ImageNet classification performance.jpg)



### Architecture and Layer Diversity

![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\layerDiversity.jpg)

```python
================================================================
Total params: 2,218,512
Trainable params: 2,218,512
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.12
Forward/backward pass size (MB): 501.67
Params size (MB): 8.46
Estimated Total Size (MB): 514.26
----------------------------------------------------------------
```



## MobileNet

在mobileNet中,会有深度可分离卷积depthwise separable convolution, 有depthwise(DW) 和 pointwise(PW)两个部分结合起来, 用来提取特征feature map

![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\convCalcu.jpg)

卷积层共4个Filter，每个Filter包含了3个Kernel，每个Kernel的大小为3×3。因此卷积层的参数数量可以用如下公式来计算：

N_std = 4 × 3 × 3 × 3 = 108

### 深度可分离卷积

- 逐通道卷积

  Depthwise Convolution的一个卷积核负责一个通道, 一个通道只被一个卷积核卷积

  一张5×5像素、三通道彩色输入图片（shape为5×5×3），Depthwise Convolution首先经过第一次卷积运算，DW完全是在二维平面内进行。卷积核的数量与上一层的通道数相同（通道和卷积核一一对应）。所以一个三通道的图像经过运算后生成了3个Feature map(如果有same padding则尺寸与输入层相同为5×5)，如下图所示。

  ![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\depthWise-1.jpg)

  其中一个Filter只包含一个为3x3的Kernel, 卷积部分的参数个数计算如下
  $$
  N_depthwise=3×3×=27
  $$
  Depthwise convolution完成后的feature map数量与输入层的通道数相同, 无法扩展Feature map. 而这种运算对输入层的每个通道独立进行卷积运算, 没有有效的利用不同通道在相同空间位置上的feature信息. 因此需要pointwise convolution来将这些feature map进行组合成新的feature map

- 逐点卷积

  pointwise convolution的运算与常规卷积运算非常相似, 它的卷积和的尺寸为1x1xM, M为上一层的通道数. 所以这里的卷积运算会将上一步的map在深度方向上进行加权组合, 生成新的feature map. 有几个卷积核就有几个输出的feature map

  ![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\pointwise.jpg)

  由于采用的1x1卷积的方式, 此步中卷积涉及到的参数个数可以计算为
  $$
  N_{pointwise}=1×1×3×4=12
  $$

- 参数对比

  1. 常规卷积个数为

     N_std=4x3x3x3=108

  2. 可分离卷积

     N_depthwise = 3 × 3 × 3 = 27
     N_pointwise = 1 × 1 × 3 × 4 = 12
     N_separable = N_depthwise + N_pointwise = 39



### MobileNetV2

mobileNetV1 的主要特点是采用深度可分离卷积替换普通卷积, 而mobileNetV2在V1的基础上引入了线性瓶颈(Linear Bottleneck)和倒残差(Inverted Residual)来提高网络的表征能力.

#### 线性瓶颈层

下图是将低维流形的ReLUctant嵌入高维空间中的例子, 原始特征通过随机矩阵T变换, 后面接ReLU层, 变化到n维空间后再通过反变换转变为原始空间. 当n=2,3时, 会导致较验证的信息丢失, 部分特征重叠起来了. 当n=15到30时, 信息丢失程度低, 但变化矩阵已经是高度非凸了. 由于非线性层会损失一部分信息, 因而使用线性瓶颈层.

![image-20210915181600560](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\ReLU-mobilenet.png)

#### 倒残差

经典的残差块(residual block)的过程是: **1x1(降维)-->3x3(卷积)-->1x1(升维)**, 但深度卷积层(Depthwise convolution layer)提取特征限制于输入特征维度, 若采用残差块, 先经过1x1的逐点卷积(pointwise convolution)操作先将输入特征图压缩(一般压缩率维0.25), 再经过深度卷积后, 提取的特征会更少. 

所以mobileNetV2是先经过1x1的逐点卷积操作将特征图的通道进行扩张, 丰富特征数量, 进而提高精度. 这一过程刚好和残差块的顺序颠倒: **1x1(升维)-->3x3(dw conv+relu)-->1x1(降维+线性变化)**

![image-20210915182321608](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\Inverted residual block.png)

#### 网络结构

瓶颈层的具体结构如下表所示. 输入通过1x1的conv + ReLU层将维度从k维增加到tk维, 之后通过3x3 conv+ReLU可分离卷积对图形进行降采样, 此时特征维度已经维tk维, 最后通过1x1 conv(无ReLU)进行降维, 维度从tk降到k维.

注意: 整个模型中除了第一个瓶颈层的t=1之外, 其他瓶颈层t=6, 即第一个瓶颈层内部并不对特征进行升维.![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\mobileNet-structure-1.png)

另外, 对于瓶颈层, 当stride=1时, 才会用elementwise的sum将输入和输出特征连接(如下左图); stride=2时, 无shortcut连接输入和输出的特征



![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\mobileNet-residual.png)

MobileNetV2的模型如下图所示, 其中t为瓶颈层内部升维的倍数, c为特征的维数, n为该瓶颈层重复的次数, s为瓶颈层第一个conv的步幅.

![image-20210915183248401](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\mobileNet-struct-2.png)

需要注意的是:

1. 当n>1时(即该瓶颈层重复的次数>1), 只在第一个瓶颈层stride为对应的s, 其他重复的瓶颈层stride均为1.
2. 只在stride=1时, 输出特征尺寸和输入特征尺寸一致, 才会使用elementwise sum将输出和输入相加
3. 当n>1时, 只咋一个瓶颈层维度为c, 其他时候channel不变.

```python
================================================================
Total params: 3,504,872
Trainable params: 3,504,872
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.12
Forward/backward pass size (MB): 1104.27
Params size (MB): 13.37
Estimated Total Size (MB): 1121.76
----------------------------------------------------------------
```



## DenseNet

![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\DenseNet.jpg)

DenseNet模型的基本思路于ResNet一致, 但是它建立的是前面所有层与后面层的密集连接(dense connection). DenseNet的一大特色是通过特征在channel上的连接来实现特征重用(feature reuse). 这些特点让DenseNet在参数和计算成本更少的情形下实现比ResNet更优的性能.

### 设计理念:

即互相连接所有的层，具体来说就是每个层都会接受其前面所有层作为其额外的输入。

### 网络结构

![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\DenseNet-structure.jpg)

在DenseBlock中，各个层的特征图大小一致，可以在channel维度上连接.DenseBlock中的非线性组合函数$H(·)$采用$BN+ReLU+3×3 Conv$​. 

另外, 与ResNet不同, 所有DenseBlock中各层卷积之后均输出k个特征图, 即得到的特征图的channel数为k(k个卷积核). k在DenseNet中称为growth rate, 这是一个超参数.一般情况下, 使用较小的k(如12), 就可以得到较佳的性能. 假定输入层的特征图的channel为k0, 那么$l$层输入的channel数位$k_0+k(l-1)$​, 因此随着层数增加, 尽管k设定较小, DenseBlock的输入也会非常大.不过这是由于特征重用所造成的, 每个层仅有k个特征是自己独立的.

![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\DenseBlock-shortcut.jpg)

由于后面的输入会非常大, DenseBlock内部可以采用bottleneck层来减少计算量, 主要是原有的结构中增加了1x1 conv, 即$BN+ReLU+1×1Conv+BN+ReLU+3×3Conv$​​, 称为Dense net-B结构.其中1x1 Conv得到的4k个特征图它起到的作用是降低特征数量, 从而提升计算效率. 

这里的4k中的4, 在代码中, 是bn_size:multiplicative factor for number of bottle neck layers

![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\DenseNet-bottleneck.jpg)

对于Transition层,它主要是连接相邻的DenseBlock, 并且降低特征图的大小. Transition层包括一个1x1 Conv和2x2 AVgPooling, 结构为:$BN+ReLU+1×1 Conv+2×2 AvgPooling$

Transition层可以起到压缩模型的作用. 假定Transition的上接DenseBlock得到的特征图channels数位m, Transition层可以产生$\theta m$个特征(通过卷积), 其中$\theta \in (0,1]$​​ 是压缩系数(compression rate). 对于使用bottleneck层的DenseBlock结构和压缩系数小于1的Transition组合结构称为DenseNet-BC。

注: 最后一层将dense block里的所有feature map都concat起来, 导致显存占用很大.

```python

================================================================
Total params: 7,978,856
Trainable params: 7,978,856
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.12
Forward/backward pass size (MB): 2229.69
Params size (MB): 30.44
Estimated Total Size (MB): 2264.25
----------------------------------------------------------------
```



## ResNet

![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\ResNet-structure.png)

```python
#ResNet50
================================================================
Total params: 25,557,032
Trainable params: 25,557,032
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.12
Forward/backward pass size (MB): 2069.00
Params size (MB): 97.49
Estimated Total Size (MB): 2170.61
----------------------------------------------------------------
```

ResNet解决的是深度神经网络的"退化"问题. 而"退化"指的是, 给网络叠加更多的层后, 性能却快速下降的情况, 训练集上的性能下降, 可以排除过拟合, BN层的引入也基本解决了plain net的梯度消失和梯度爆炸问题.

两种解决思路, 以实调整求解方法, 比如更好的初始化, 更好的梯度下降算法等; 另一种是调整模型结构, 让模型更易于优化, 而改变模型结构实际上是改变error surface的形态.

### 残差结构

- Residual Block

![image-20210924104041699](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\image-20210924104041699-16324512429035.png)

一个残差块有2条路径 $F(x)$ 和 $x, F(x)$ 路径拟合残差，不妨称之为残差路径, $x$路径为identity mapping恒等映射，称之为" shortcut" 。图中的 $\oplus$ 为element-wise addition，要求参与运算的 $F(x)$ 和 $x$ 的尺寸要相同。所 以，随之而来的问题是.

- BasicBlock+Bottleneck

![image-20210924103236184](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\image-20210924103236184-16324507643293.png)
$$
H(x)=F(x)+x
$$

```python

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out
```



### 网络结构

![image-20210924103151529](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\image-20210924103151529-16324507124102.png)

![image-20210924103522550](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\image-20210924103522550-16324509238794.png)

### error surface对比

ResNet深度增加到1000层以上, 也没有发生"退化", 可见Residual Block是非常有效的. ResNet的动机在于认为拟合残差比直接拟合潜在映射更容易优化, 下面通过绘制error surface直观感受shortcut路径的作用,来自[loss visualization](http://www.telesens.co/loss-landscape-viz/viewer.html#)

![http://www.telesens.co/loss-landscape-viz/viewer.html](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\38Kd2t.png)

可以发现: 

- ResNet-20 (no short) 浅层plain net的error surface还没有很复杂, 优化也会很困难, 但是增加到56层后, 复杂程度极度上升. 对于plain net, 随着深度增加, error surface迅速"恶化"
- 引入short cut后, error suface变得平滑很多, 梯度的可预测性变得更好, 更容易优化

### Residual Block 的分析与改进

论文[Identity Mapping in Deep Residual Network](https://arxiv.org/pdf/1603.05027.pdf) 进一步研究ResNet, 通过ResNet反向传播的理论分析以及调整Residual Block的结构, 得到新的结构

![https://arxiv.org/abs/1603.05027](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\3G0uVJ.png)

注意, 这里的视角与之前不同, 这里将short cut路径视为主干路径, 将残差路径视为旁路. 

新提出的Residual Block结构, 具有更好的泛化能力, 能更好的避免"退化", 堆叠大于1000层后, 性能仍在变好. 具体变化在于

- 通过保持short cut路径的"纯净", 可以让信息在前向传播和反向传播中平滑传递, 这点十分重要. 为此, 如无必要, 不要引入1x1卷积操作, 同时将上图灰色路径上ReLU移到$F(x)$路径上.
- 在残差路径上, 将BN和ReLU统一放在weight前作为pre-activation, 获得了"Ease of optimization"以及 "Reducing overfitting"的效果.

对于残差路径的改进, 作者进行不同对比实验, 最终得到了将BN和ReLU统一放在weight前的full preactivation结构

![https://arxiv.org/abs/1603.05027](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\3GcfnP.png)

- [ResNet 详解与分析](https://www.cnblogs.com/shine-lee/p/12363488.html)

## VGG

![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\VGG16-structure.jpg)

```python
#VGG16
================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.12
Forward/backward pass size (MB): 1566.19
Params size (MB): 527.79
Estimated Total Size (MB): 2098.10
----------------------------------------------------------------
```



## SequeezeNet| 轻量级深层神经网络

论文链接：http://arxiv.org/abs/1602.07360
 代码链接（Caffe）：https://github.com/DeepScale/SqueezeNet
 代码链接（Keras）：https://github.com/rcmalli/keras-squeezenet

### SqueezeNet网络结构

其核心结构FireModule的组合形式. 作图是SqueezeNet的整体结构, 其余两图是ResNet网络的shortcut引入所构建的网络.

![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\SqueezeNet.jpg)

上图左图网络中各层的滤波器参数具体如下

![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\SqueezeNet_filter.jpg)

### FireModule详解

![img](E:\kuisu\typora\torchvision\classifier 各种分类模型学习笔记.assets\FireModule.jpg)

如上图所示, FireModule由squeeze模块和expand模块. 该模块的主要两个特性.

1. squeeze模块: 利用1x1卷积进行将为(所以如图中的16<128)
2. expand模块: 利用1x1卷积+3x3卷积组合升维

整个网络还有一个特性:

3. 将pooling采样操作延后, 可以给卷积层提供更大的激活图: 更大的激活图保留了更多的信息, 可以提提供更高的分类准确率.

其中, 1), 2) 可以显著减少参数数量, 3)可以再参数数量受限的情况下提高准确率.

```python
#SqueezeNet
================================================================
Total params: 1,235,496
Trainable params: 1,235,496
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.12
Forward/backward pass size (MB): 403.69
Params size (MB): 4.71
Estimated Total Size (MB): 412.52
----------------------------------------------------------------
```

