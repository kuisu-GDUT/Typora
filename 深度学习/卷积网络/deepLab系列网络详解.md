# deepLab系列网络详解

## deepLab-V1

《Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFS》(http://arxiv.org/pdf/1412.7062v3.pdf)

DeepLab是结合了深度卷积网络(DCNNs)和概率图模型(DenseCRFs)的方法, 在实验中发现DCNNs做语义分割时精度不够的问题, 根本原因时DCNNs的高级特征的平移不变性.

由于卷积神经网络在提取特征时, 会将输入图像逐渐缩小, featuremap变小形成高级别的特征对分割任务并不适用, deepLab采用空洞卷积替换传统的卷积和fully connected CRF. 为了利用已经训练好的VGG模型进行fine-tunning, 又想改变网络结构得到更加dense的score map而引入空洞卷积

1. 空洞卷积的引入

   VGG16的原始模型, 卷积层的卷积核大小统一为3x3, 步长为1, 最大池化层的池化窗口为2x2, 步长为2.

   改进是使用1x1的卷积层代替FC层, 那么旧变成了全卷积网络, 输出得到的是得分图, 也可理解成概率图. 将pool4和pool5步长由2改为1, 这样在原本FC7的位置, VGG网络总的步长由原来的32变为8.这样改变的原因是为了获得更稠密(dense)的score map. 

2. Fully connected CRF

   CRF 是conditional Random Field(条件随机场), 在图像处理领域的作用是平滑处理, 在针对某个位置的像素值处理时, 会综合考虑周围像素的值, 采用Fully connected CRF可以综合考虑全局信息, 恢复详细的局部结构, 如精确图像的轮廓. CRF几乎可以用于所有的分割任务中图像进度的提高.

![image-20211008144334114](E:\kuisu\typora\深度学习资料\卷积网络\deepLab系列网络详解.assets\fully connected CRF.png)

第一列时原图和Ground Truth; 第二列时DCNN的输出, 上面时得分图(score map), 下面时置信度图(Belief map); 最后一个DCNN层的输出作用域CRF的输入,后面三列分别时CRF迭代1,2,10次后的得分图和置信图.

![image-20211008144640299](E:\kuisu\typora\深度学习资料\卷积网络\deepLab系列网络详解.assets\workflow.png)

CRF的能量函数
$$
E(x)=\sum_i \theta_i(x_i)+\sum_{ij} \theta_{ij}(x_i, x_j)\\
\theta_i(x_i)=-log P(x_i)\\
\theta_{ij}(x_i,x_j)=\mu(x_i,x_j)\sum_{m=1}^K w_m *k^m(f_i,f_j)\\
\mu(x_i,x_j)=1 \text{ if }x_i≠ x_j
$$
最右边为高斯核函数:$k^m(f_i,f_j)$

![image-20211008145706721](E:\kuisu\typora\深度学习资料\卷积网络\deepLab系列网络详解.assets\gaosi_kernel_function.png)

此核函数主要由两个像素点的`位置`和`颜色`决定, 位置为主, 颜色为辅. 

- 网络结构
- ![image-20211010141310551](E:\kuisu\typora\深度学习资料\卷积网络\deepLab系列网络详解.assets\structural-deeplabv1.png)

- [分割网络](https://cloud.tencent.com/developer/article/1632442)

## DeepLab-V2

DeepLab-V2相对于v1最大的改动是增加了受SPP(spacial pyramid pooling)启发得来的ASPP(Atrous Spacial Pyramid Pooling), 在模型最后进行像素分类之前, 增加一个类似于inception的结构, 包含不同rate(空间间隔)的Atrous Conv(空洞卷积), 增强模型识别不同尺寸的统一物体的能力.

![image-20211008160427995](E:\kuisu\typora\深度学习资料\卷积网络\deepLab系列网络详解.assets\image-20211008160427995-16336802691402-16336802892123.png)

![image-20211008160238739](E:\kuisu\typora\深度学习资料\卷积网络\deepLab系列网络详解.assets\deepLab-v2-ASPP.png)

![image-20211008160343813](E:\kuisu\typora\深度学习资料\卷积网络\deepLab系列网络详解.assets\deepLab-v2-ASPP-compare.png)

## DeepLab-V3

DeepLabV3的主要变化

1. 使用Multi-Grid策略, 即在模型后端多加几层不同rate的空洞卷积:

   ![img](E:\kuisu\typora\深度学习资料\卷积网络\deepLab系列网络详解.assets\deepLabV3-Multi-Grid.jpg)

2. 将batch normalization 加入到ASPP模块

3. 具有不同atrous rates的ASPP能够有效的捕获多尺度信息, 不过, 论文发现, 随着sampling rate增加, 有效filter特征权重(即有效特征区域, 而不是补零区域的权重)的数量会变小, 极端情况下, 当空洞卷积的rate和feature map的大小一致时, 3x3卷积会退化成1x1

   ![img](E:\kuisu\typora\深度学习资料\卷积网络\deepLab系列网络详解.assets\atrous rate.jpg)

   为保留较大视野的空洞卷积的同时解决这个问题, DeepLabV3的ASPP加入了全局池化层+conv1x1+双线性插值上采样的模块

   

![img](E:\kuisu\typora\深度学习资料\卷积网络\deepLab系列网络详解.assets\ASPP.jpg)

## DeepLab-V3+

V3+最大的改进是将DeepLab的DCNN部分看做Encoder, 将DCNN输出的特征图上采样成原图大小的部分看做Decoder, 构成Encoder+Decoder体系, 双线性插值上采样便是一个简单的Decoder, 而强化Decoder便可使模型整体在图像语义分割边缘部分取得良好的结果

![img](E:\kuisu\typora\深度学习资料\卷积网络\deepLab系列网络详解.assets\DeepLab-V3+.jpg)

具体来说, DeepLabV3+在stride=16的DeepLabV3模型输出上采样4x后, 将DCNN中0.25x的输出使用1x1的卷积降维后与之连接(concat)再使用3x3卷积处理后双线性插值上采样4倍后,得到相对于DeepLabV3更精细的结果

![img](E:\kuisu\typora\深度学习资料\卷积网络\deepLab系列网络详解.assets\DeepLabv3+ extends.jpg)

deepLabV3+的其他改进

1. 借鉴mobileNet, 使用Depth-wise空洞卷积+1x1卷积

![img](E:\kuisu\typora\深度学习资料\卷积网络\deepLab系列网络详解.assets\Atrous depthwise conv.jpg)

2. 使用修改过的Xception:

![img](E:\kuisu\typora\深度学习资料\卷积网络\deepLab系列网络详解.assets\Xception-edit.jpg)

## Xception详解

