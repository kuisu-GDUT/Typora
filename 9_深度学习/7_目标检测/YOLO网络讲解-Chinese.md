## YOLO-V1

1. 统一网络: YOLO没有显示求取region proposals的过程. Faster R-CNN中尽管RPN与fast rcnn共享卷积层, 但是模型在训练过程中, 需要反复训练RPN网络, 相对于R-CNN.
2. YOLO统一为一个回归问题, 而Faster R-CNN将检测结果分为两部分求解: 分类, 位置回归

- 论文地址: https://arxiv.org/abs/1506.02640

- 官方代码: https://github.com/pjreddie/darknet
- 核心思想: 利用整张图作为网络的输入, 直接在输出层回归bounding box的位置和boundingbox所属类别

## 实现方法

- 将一幅图像分成SxS个网格(grid cell), 如果某个object的中心落在这个网络中, 这个网络就负责预测这个object

  ![image-20211129091105197](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\image-20211129091105197-16381482666701.png)

- 每个网格要预测B个bound ing box, 每个bound ing box除了要回归自身的位置之外, 还要附带预测一个confidence值. 这个confidence代表了所预测box中含有object的置信度和这个box预测的有多准确两重信息
  $$
  Pr(Object)*IOU_{pred}^{truth}
  $$
  表达式含义: 如果有object落在一个grid cell里, 则第一项取1, 否则取0. 第二项是预测的bounding box和实际groundtruth之间的IOU值.

  - 每个bounding box要预测(x,y,w,h)和confidence共5个值, 每个网络还要预测一个类别信息C. 则SxS个网格, 输出就是SxSx(5*B+C)的一个tensor.

    注: class信息是针对每个网格的, confidence是针对每个bounding box的.

## 实例

在PASCAL VOC中, 图像输入为448x448, S=7, B=2, 共有20个类别(C=20), 输出就是7x7x(2x5+20)的tensor. 整个网络结构如图所示

![image-20211129092059854](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\image-20211129092059854-16381488607922.png)

在test的时候, 每个网格预测的class信息和bounding box预测的confidence信息相乘, 就得到每个bounding box的class-specific confidence score, 得到每个box的class-specific confidence score之后, 设置置信度阈值, 滤掉得分低的boxes, 对保留的boxes进行NMS处理, 就得到最终的检测结果.
$$
Pr(Class_i|Object)*Pr(Object)*IOU_{pred}^{truth}=Pr(Class_i)*IOU^{truth}_{pred}
$$
表达式含义: 等式左边第一项就是每个网格预测的类别信息, 第二三项为每个bounding box预测confidence. 这个乘积即encode了预测的box属于某一类的概率, 也有该box准确度的信息.

注: 

1. 由于输出层为全连接层, 因此在检测时, YOLOv1模型的输入只支持与训练图像相同的输入分辨率
2. 虽然每个格子可以预测B个bounding box, 但是最终只选择IOU最高的bounding box作为物体检测输出, 即每个格子最多只预测出一个物体. 当物体占画面比较小, 如图像中包含鸟群时, 每个格子包含多个物体, 但却只能检测出其中一个. 

## 损失函数

每个grid有30维, 其中8维是回归box的坐标, 2维是box的confidence, 另外20维是类别. 其中坐标的x,y用对应网格的offset归一化到0-1之间, w,h用图像的width和height归一化到0-1之间. 在实现中, 最主要的就是怎么设计损失函数, 让这三个方面得到很好的平衡. 作者采用最简单粗暴的sum-squared error loss

潜在的问题:

1. 8维的localization error和20维的classification error同等重要显然是不合理的
2. 如果一个网格中没有object(一幅图中, 这种没有object的网格占多数), 那么就会将这些网格中的box的confidence push到0, 相比较少的object的网格, 这种做法是overpowering的, 这回导致网络不稳定

YOLO v1在对不同大小box的预测中, 相比于大box预测偏一点, 小box预测偏一点肯定不能忍受. 而sum-square error loss中对不同样的偏移loss是一样的. 维缓和这个问题, 作者采用将box的width和height取平方根代替原本的height和width. 如下图, 小的box的横轴值较小, 发生偏移时, 反应到y轴上相比大box要大

![img](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\20160317163247283)

一个网格预测多个box, 希望每个box predictor专门负责预测某个object. 具体做法就是看当前预测的box于ground truth box中哪个IOU大, 就负责哪个. 这种做法称作box predictor的specialization.
$$
\begin{gathered}
\lambda_{\text {coord }} \sum_{i=0}^{S^{2}} \sum_{j=0}^{B} \mathbb{1}_{i j}^{\text {obj }}\left[\left(x_{i}-\hat{x}_{i}\right)^{2}+\left(y_{i}-\hat{y}_{i}\right)^{2}\right] \\
+\lambda_{\text {coord }} \sum_{i=0}^{S^{2}} \sum_{j=0}^{B} \mathbb{1}_{i j}^{\text {obj }}\left[\left(\sqrt{w_{i}}-\sqrt{\hat{w}_{i}}\right)^{2}+\left(\sqrt{h_{i}}-\sqrt{\hat{h}_{i}}\right)^{2}\right] \\
+\sum_{i=0}^{S^{2}} \sum_{j=0}^{B} \mathbb{1}_{i j}^{\text {obj }}\left(C_{i}-\hat{C}_{i}\right)^{2} \\
+\lambda_{\text {noobj }} \sum_{i=0}^{S^{2}} \sum_{j=0}^{B} \mathbb{1}_{i j}^{\text {noobj }}\left(C_{i}-\hat{C}_{i}\right)^{2} \\
\quad+\sum_{i=0}^{S^{2}} \mathbb{1}_{i}^{\text {obj }} \sum_{c \in \text { classes }}\left(p_{i}(c)-\hat{p}_{i}(c)\right)^{2}
\end{gathered}
$$
在YOLO v1的损失函数中:

1. 只有当某个网格中有object的时候才对classification error进行惩罚
2. 只有当某个box predictor对某个ground truth box负责时, 才会对box的coordinate error进行惩罚, 而对哪个ground truth box负责就看其预测值和ground truth box的IOU是不是在那个cell的所有box中最大.

注: 

1. YOLO v1模型训练依赖于物体识别标注数据, 因此, 对于非常规的物体形状或比例, 其检查效果不理想
2. YOLO v1采用多个下采样层, 网络学到的物体特征并不精细, 因此也会影响检测效果. 
3. YOLO v1的loss函数中, 达吾提IOU误差和小物体IOU误差对网络训练中的loss贡献值接近, 因此, 对于小物体, 小的IOU误差也会对网络优化过程造成很大的影响, 从而降低了物体检测的定位准确性.

YOLO 缺点

- YOLO对相互靠的很近的物体和很小的群体检测效果不好, 这是因为一个网格中只预测了两个框, 并且只属于一类. 
- 同一类物体出现的新的不常见的长宽比和其他情况时, 泛化能力偏弱
- 由于损失函数的问题, 定位误差时影响检测效果的主要原因.

### 代码示例

- 将bbox信息转换成YOLOV1网络所需的标签格式

```python
def convert_bbox2labels(bbox):
    """
    将bbox的(cls, x,y,w,h)数据转为训练时方便计算Loss的数据形式(7,7,5*B+cls_num), 注意:输入的bbox的信息是(cx,cy,w,h)格式, 转为labels后, bbox的信息转换为(px,py,w,h)格式
    """
    gridsize = 1.0/7
    labels = np.zeros((7,7,5*NUM_BBOX+len(CLASSES)))#此处需要根据不同数据集的类别个数进行修改
    for i in range(len(bbox)//5):
        gridx = int(bbox[i*5+1]//gridsize)#当前bbox中心落在第gridx个网格,列
        gridx = int(bbox[i*5+2]//gridsize)#当前bbox中心落在第gridy个网格,行
        #(bbox中心坐标-网格左上角点的坐标)/网格大小==>bbox中心点的相对位置
        gridpx = bbox[i*5+1]/gridsize-gridx
        gridpy = bbox[i*5+2]/gridsize-gridy
        #将gridy行, gridx列的网格设置为当前ground truth的预测, 置信度和对应类别概率均设置为1
        labels[gridy,gridx,0:5]=np.array([gridpx,gridpy,bbox[i*5+3],bbox[i*5+4],1])
        labels[gridy,gridx,5:10]=np.array([gridpx,gridpy,bbox[i*5+3],bbox[i*5+4],1])
        labels[gridy,gridx,10+int(bbox[i*5])]=1
    return labels
```

- loss函数设计

  未来方便Loss计算, 我们将网络输出的一维1470数据reshape成7x7x30的数据格式, 并且, 根据上下文Dataset类的实现可知, 提取的label(样本标签)也是7x7x30. 但需要注意, label数据经过pytorch的toTensor()函数转换后, 数据会变成batchsizex30x7x7, 所以网络的输出也应当改成batchsizex30x7x7

  ```python
  class Loss_yolov1(nn.Module):
      def __init__(self):
          super(Loss_yolov1,self).__init__()
      def forward(self,pred,labels):
          '''
          pred: (batchsize,30,7,7)的网络输出数据
          labels: (batchsize,30,7,7)的样本标签数据
          return: 当前批次样本的平均损失
          '''
          num_gridx, num_gridy = labels.size()[-2:]#划分网格数量
          num_b = 2#每个网格的bbox数量
          num_cls = 20#类别数量
          noobj_confi_loss = 0.#不含目标的网络损失(只有置信度损失)
          coor_loss = 0.#含有目标的bbox的坐标损失
          obj_confi_loss = 0.#含有目标的bbox的置信度损失
          class_loss = #含有目标的网格的类别损失
          n_batch = labels.size()[0]
          
          #可以考虑矩阵运算进行优化, 为了准确起见, 目前用循环
          for i in range(n_batch):#batchsize循环
              for n in range(7):#x方向网格循环
                  for m in range(7):#y方向网格循环
                      if labels[i,4,m,n]==1:#如果包含物体
                          #将数据(px,py,w,h)转化成(x1,y1,x2,y2)
                          #先将px,py转换为cx,cy, 即相对网格的位置转换为标准化后实际的bbox中位置cx,cy, 然后再利用(cx-w/2, cy-h/2, cx+w/2,cy+h/2)转换为xyxy形式, 用于计算iou
                          bbox1_pred_xyxy = ((pred[i,0,m,n]+n)/num_gridx-pred[i,2,m,n]/2,
                                             (pred[i,1,m,n]+m)/num_gridy-pred[i,3,m,n]/2,
                                             (pred[i,0,m,n]+n)/num_grdix+pred[i,2,m,n]/2,
                                             (pred[i,1,m,n]+m)/num_gridy+pred[i,3,m,n]/2)
                          bbox2_pred_xyxy = ((pred[i,5,m,n]+n)/num_gridx-pred[i,7,m,n]/2,
                                             (pred[i,6,m,n]+m)/num_gridy-pred[i,8,m,n]/2,
                                             (pred[i,5,m,n]+n)/num_grdix+pred[i,7,m,n]/2,
                                             (pred[i,6,m,n]+m)/num_gridy+pred[i,8,m,n]/2)
                          bbox_gt_xyxy = ((labels[i,0,m,n]+n)/num_gridx-labels[i,2,m,n]/2,
                                         (labels[i,1,m,n]+n)/num_gridx-labels[i,3,m,n]/2,
                                         (labels[i,0,m,n]+n)/num_gridx+labels[i,2,m,n]/2,
                                         (labels[i,1,m,n]+n)/num_gridx-labels[i,3,m,n]/2)
                          iou1 = calculate_iou(bbox1_pred_xyxy,bbox_gt_xyxy)
                          iou2 = calculate_iou(bbox2_pred_xyxy,bbox_gt_xyxy)
                          #选择iou大的bbox作为负责物体
                          if iou1>=iou2:
                              coor_loss += 5*(torch.sum((pred[i,0:2,m,n]-
                                                         labels[i,0:2,m,n])**2)+
                                              torch.sum((pred[i,2:4,m,n].sqrt()-
                                                         labels[i,2:4,m,n].sqrt())**2))
                              obj_confi_loss = obj_confi_loss+(pred[i,4,m,n]-iou1)**2
                              #iou比较的bbox不负责预测物体, 因此confidence loss算在noobj中,注意,对于标签的置信度应该是iou2
                              noobj_confi_loss = noobj_confi_loss+0.5*((pred[i,9,m,n]-iou2)**2)
                          else:
                              coor_loss += 5*(torch.sum((pred[i,5:7,m,n]-
                                                         labels[i,5:7,m,n])**2)+
                                              torch.sum((pred[i,7:9,m,n].sqrt()-
                                                         labels[i,7:9,m,n].sqrt())**2))
                              obj_confi_loss = obj_confi_loss+(pred[i,9,m,n]-iou2)**2
                              #iou较小的bbox不负责预测物体,因此confidence loss算在nobj中
                              noobj_confi_loss = noobj_confi_loss+0.5*((pred[i,4,m,n]-iou1)**2)
                          class_loss = class_loss+torch.sum((pred[i,10:,m,n]-labels[i,10:,m,n])**2)
                      else:#如果不包含物体
                          noobj_confi_loss = noobj_confi_loss+0.5*torch.sum(pred[i,[4,9],m,n]**2)
  		loss=coor_loss+obj_confi_loss+noobj_confi_loss+class_loss
          return loss/n_batch
                          
                          
  ```

- 网络结构

  由于原论文是采用自己设计的20层卷积层现在ImageNet上训练一周, 完成特征提取部分的训练. Yolo v1的前20层是用来特征提取的, 也就是随便替换为一个分类网络(除去最后的全连接层)其实都行. 因此, 这里使用ResNet34的网络作为特征提取部分. 这样做的好处是, pytorch的torchvision中提供了ResNet34的预训练模型, 训练集也是ImageNet, 然后除去ResNet34的最后两层, 再连接上YOLOv1的最后4卷积层和两个全连接层, 作为我们训练的网络结构

  此外, 还进行一些小的调整, 比如最后增加一个sigmoid层, 以及再卷积层后增加BN层

  ```python
  import torchvision.model as tvmodel
  import torch.nn as nn
  import torch
  
  class YOLOv1_resnet(nn.Module):
      def __init__(self):
          super(YOLOv1_resnet,self).__init__()
          resnet = tvmodel.resnet34(pretrained=True)#调用torchvision里的resnet34预训练模型
          resnet_out_channel=resnet.fc.in_features#记录resnet全连接层之前的网络输出通道数,方便连入后续卷积网络中
          self.resnet = nn.Sequential(*list(resnet.children())[:-2])#去除resnet的最后两层
          #以下是YOLOv1的最后四个卷积层
          self.Conv_layers = nn.Sequential(
              nn.Conv2d(resnet_out_channel,1024,padding=1),
              nn.BatchNorma2d(1024),#为了加快训练, 这里增加BN
              nn.LeakyReLU(),
              nn.Conv2d(1024,1024,3,stride=2,padding=1),
              bb.BatchNorm2d(1024),
              nn.LeakyReLU(),
              nn.Conv2d(1024,1024,3,padding=1),
              nn.BatchNorm2d(1024),
              nn.LeakyReLU(),
              nn.Conv2d(1024,1024,3,padding=1),
              nn.BatchNorm2d(1024),
              nn.LeakyReLU()
          )
          
          #以下是YOLOv1的最后两个全连接层
          self.Conn_layers = nn.Sequential(
          	nn.Linear(7*7*1024,4096),
              nn.LeakyReLU(),
              nn.Linear(4096,7*7*30),
              nn.Sigmoid()#增加sigmoid函数是为了将输出全部映射到(0,1)之间,因为如果出现负数或太大的数, 后续计算loss会很麻烦
          )
      def forward(self,input):
          input = self.resnet(input)
          input = self.Conv_layers(input)
          input = input.view(input.size()[0,-1])
          input=self.Conn_layers(input)
          return input.reshape(-1,(5*NUM_BBOX+len(CLASSES)),7)
  ```

  



# YOLO V2详解

- 论文地址: https://arxiv.org/abs/1612.08242
- 官方代码: http://pjreddie.com/darknet/yolo/

YOLO v1有很多缺点, 作者希望改进的方向是: 改善recall, 提升定位的准确度, 同时保持分类的准确度. 具体改进如下表

![img](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNzE2OTQ0,size_16,color_FFFFFF,t_70)

## YOLO v2改进点

### Batch Mormalization

批量归一化有助于解决反向传播过程中的梯度消失和梯度爆炸问题, 降低对一些超参数(如学习率, 网络参数的大小范围, 激活函数的选择)的敏感性, 并且每个batch分别进行归一化的时候, 起到了一定的正则化效果(YOLO v2不再使用dropout), 从而获得更好的收敛速度效果. 

使用BN对网络进行优化, 让网络提高了收敛性, 同时还消除了对其他形式的正则化(regularization)的依赖. 通过对YOLO v2的每一个卷积层增加Batch Normalize, 最终使得mAP提高2%, 同时还使model正则化. 使用BN可以从model中去掉DropOut, 而不会产生过拟合. 

### High resolution classifier

用于图像分类的训练样本很多, 而标注了边框的用于训练目标检测的样本相比而言就少了很多, 因为标注边框的人工成本比较高. 所以目标检测模型通常都先用图像分类样本训练卷积层, 提取图像特征, 但这引出另一个问题, 就是图像分类样本的分辨率不是很高. 所以YOLO v1使用ImageNet的图像分类样本采用224x224作为输入, 来训练CNN卷积层. 然后再训练目标检测时, 检测用的图像样本采用更高分辨率的448*448图像作为输入, 但这样不一致的输入分辨率肯定会对模型性能有一定影响.

所以YOLO v2在采用224*224图像进行分类模型预训练后, 再采用448x448高分辨率样本对分类模型进行微调(10epoch), 时网络特征逐渐适应448x448的分辨率. 然后再使用448x448的检测样本进行训练, 缓解了分辨率突然切换造成的影响, 最终通过使用高分辨率, mAP提升了4%

### Convolution with anchor boxes

YOLO v1包含有全连接层, 从而能直接预测Bounding Boxes的坐标值. Faster R-CNN算法只有卷积层和Region Proposal Network来预测Anchor Box的偏移值与置信度, 而不是直接预测坐标值, YOLO v2作者发现通过预测偏移量而不是坐标值能够简化问题, 让网络学习起来更容易

借鉴Faster RCNN的做法, YOLO v2也尝试采用先验框(anchor), 在每个grid预先设定一组不同大小和宽高比的边框, 来覆盖整个图像的不同位置和多种尺度, 这些先验框作为预定义的候选区在神经网络中检测其中是否存在对象, 以及微调边框的位置.

之前Yolo v1并没有采用先验框, 并且每个grid只预测两个bounding box, 也就是整个图像只有98个bounding box. YOLO v2如果每个grid采用9个先验框, 总共有13x13x9=1521个先验框, 所以最终YOLO v2去掉了全连接层, 使用Anchor Boxes来预测Bounding Boxes, 作者去掉网络中一个Pooling层, 这让卷积层的输出能有更高的分辨率, 同时对网络结构进行收缩让其运行在418x416而不是448x448.

由于图像中的物体都倾向于出现在图片的中心位置, 特别时那种比较大的物体, 所以有一个单独位于物体中心的位置用于预测这些物体. YOLO v2的卷积层采用32倍下采样图像, 所以通过选择416x416用作输入尺寸, 最终能输出一个13*13的feature map. 使用anchor box会让准确度稍微下降, 但用了它能让YOLO v2能预测出大于1000个框, 同时recall达到88%, mAP达到69.2%

### Direct location prediction

用Anchor Box的方法, 会让model变得不稳定, 尤其在最初几次迭代的时候, 大多数不稳定因素产生自预测Box的(x,y)位置时候, 按照之前YOLO v1的方法, 网络不会预测偏移量, 而不是根据YOLO V1中的网格单元的位置来直接预测坐标. 这就让Ground Truth的值介于0-1之间. 而为了让网络的结果能落到这一范围内, 网络使用一个Logistic Activation来对网络预测结果进行限制, 让结果介于0-1之间. 网络在每一个网络单元中预测出5个Bounding Box, 每个Bounding Boxes有5个坐标值tx, ty, tw, th, to, 他们的关系见下图, 假设一个网络单元对于图片左上角的偏移量是cx, cy, Boudning Boxes prior的宽度和高度是pw, ph, 那么预测的结果见下图右面的公式:

![img](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\DirectLocationPrediction.png)

## Fine-Grained Features

目标检测面临的一个问题是图像中的需要检测的目标会有大有小, 输入图像经过多层网络提取特征, 最后输出的特征图中(比如YOLO v2中输入416x416经过卷积网络下采样后输出13x13), 较小的对象可能特征已经不明显了. 为了更好的检测出一些比较小的对象, 最后输出的特征图需要保留一些更细节的信息. 于是YOLO v2引入一种称为passthrough层的方法在特征图中保留一些细节信息. 具体来说, 就是在最后一个pooling之前, 特征图的大小是26x26x512, 将其1拆4, 直接传递(passthrough)到pooling后(并且又经过一组卷积)的特征图, 两者叠加到一起作为输出的特征图. 

![img](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\Fine-Grained Feature.png)

具体怎样将一个特征图拆成4个特征图, 见下图, 图中示例的是一个4x4拆成4个2x2, 因为深度不变, 所以没画出来.

![img](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\Fine-Grained Feature_pol.png)

### Multi-Scale Training

作者希望YOLO v2能健壮的运行于不同尺寸的图片上, 所以把这一想法用于训练model中, 区别于之前的补全图片的尺寸的方法, YOLO v2每迭代几次都会改变网络参数. 每10个Batch, 网络会随机地选择一个新的图片尺寸, 由于使用了下采样参数是32, 所以不同尺寸大小也选择32的倍数{320, 352, ..., 608}, 网络会自动改变尺寸, 并继续训练的过程, 这一政策让网络在不同的输入尺寸上都能达到一个很好的预测效果, 统一网络能在不同分辨率上进行检测. 当输入图片尺寸比较小的时候, 运行比较快, 输入图片尺寸比较大的时候, 精度高, 所以可以在YOLO v2的速度和精度上进行权衡.

![img](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\multiScaleTraining.png)

![img](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\multiScaleTraining-1.png)

### YOLOv2 Faster

YOLO v1的backbone使用的是GoogleNet, 速度比VGG-16快, YOLOv1完成一次前向过程中只用8.52 billion运算, 而VGG-16要30.69 billion, 但YOLO v1精度稍低于VGG-16

### DarkNet19

YOLO v2基于一个新的分类model, 有点类似于VGG. YOLO v2 使用3x3 filter, 每次pooling之后都增加一倍channels数量. YOLO v2使用Global Average Pooling, 使用Batch Normalization来让训练更稳定, 加速收敛, 使model规范化. 最终的model-DarkNet19, 有19个卷积层和5个maxpooling层, 处理一张图片只需要5.58 billion次运算, 在ImageNet上达到72.9% top-1精确的, 91.2% top-5精确度

![img](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\Darknet19.png)

### Training for classification

网络训练在ImageNet 1000类分类数据上训练了160epochs, 使用随机梯度下降, 初始学习率为0.1, polynomial rate decay with a power of 4, weight decay of 0.0005 and momentum of 0.9. 训练期间使用标注的数据扩大方法, 随机裁剪, 旋转, 变换颜色(hue), 变换饱和度(saturation), 变换曝光度(exposure shifts). 在训练时, 把整个网络在更大的448x448分辨率上Fine Turning 10个epoches, 初始学习率设置为0.001, 这种网络达到76.5% top-1精确度, 93.3% top-5精确度.

### Training for detection

网络去掉了最后一个卷积层, 而加上了三个3x3卷积层, 每个卷积层1024个filters, 每个卷积层紧接着一个1x1卷积层. 对于VOC数据, 网络预测出每个网络单元预测5个Bounding Boxes, 每个Bounding Boxes预测5个坐标和20个类, 所以一共125个Filters, 增加Passthrough层来获取前面层的细粒度信息, 网络训练了160 epoches, 初始学习率0.001, 数据扩大方法相同, 对COCO于VOC数据集的训练对策相同.

## YOLO V3

论文地址: https://pjreddie.com/media/files/papers/YOLOv3.pdf 

![img](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\YOLO-V3.png)

- yolov3 上保留的东西
  1. "分而治之", 从yolo_v1开始, yolo算法就是通过划分单元格来做检测, 只是划分的数量不一样.
  2. 采用"leaky ReLU"作为激活函数
  3. 端到端进行训练, 一个loss function搞定训练, 只需关注输入端到输出端
  4. 从yolo_v2开始, yolo就用batch normalization作为正则化, 加速收敛和避免过拟合的方法, 把BN层和leaky relu层接到每一层卷积之后.
  5. 多尺度训练, 

![img](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\yolov3_structure.png)

> DBL: YOLO_v3的基本组件, 就是卷积+BN+Leaky relu. 对于V3来说, BN和leaky relu已经是和卷积层不可分离的部分
>
> resn: n代表数字, res1, res2, ..., res8, 表示这个res_block里含有多少个res_unit. 
>
> concat: 张量拼接. 将darknet中间层和后面的某一层的上采样进行拼接. 拼接的操作和残差层的add的操作是不一样的, 拼接会扩充张量的维度, 而add只是直接相加,不会导致张量维度的改变.

### backbone

整个v3结构里面, 没有池化层和全连接层. 前向传播过程中, 张量的尺寸变化是通过改变卷积核的步长来实现的. 再yolov3中会尽力5次缩小, 会将特征图缩小到原输入尺寸的$1/2^5$, 即$1/32$. 输入为416x416, 则输出为13x13

![img](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\yolov3_baseNet.png)

yolo_v2中对于前向过程中张量尺寸变化, 都是通过最大池化来进行, 一共有5次, 而V3是通过卷积核增大步长来进行, 也是5次. 

![这里写图片描述](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\backboen_compare.png)

tiny-darknet作为backbone可以代替darknet-53, 再官方代码里面用一行代码可以实现切换backbone. 搭用tiny-darknet的yolo, 具备轻量和高速两个特点.

![img](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\tinyDarknet.png)

### output

![img](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\yolov3_output.png)

yolo v3输出3个不同尺度的feature map( prediction across scales). 

如上图, y1, y2, y3的深度都是255, 变成的规律是13:26:52

对于coco类检测而言, 有80个种类, 所以每个box需要有(x,y,w,h,confidence)五个基本参数, 然后还有80个类别的概率, 每个cell设定3个初始的anchor, 所以`3*(5+80)=255`. 

V3采用上采样的方法实现多尺度的feature  map, 其`concat`连接的两个张量是具有一样尺度(两处拼接分别是26x26尺度拼接和52x52尺度拼接, 通过(2,2)上采样来保证concat拼接的张量尺度相同) . 

在SSD中是直接采用backbone中间层的处理结果作为feature map的输出

YOLO是后面网络层的上采样结果进行一个拼接之后的处理结果作为feature map

### YOLOLayer code

```python
class YOLOLayer(nn.Module):
    '''对YOLO的输出进行处理'''
    def __init__(self, anchors, nc, img_size, stride):
        super(YOLOLayer,self).__init__()
        self.anchors=torch.Tensor(anchors)
        self.stride = stride#layer stride 特征图上一步对应原图上的步距[32,16,8]
        self.na = len(anchors)#number of anchors (3)
        self.nv = nv #number of class (80)
        self.no = nc+5#number of outputs (25:x,y,w,h,obj,cls1,cls2...)
        self.nx,self.ny, self.ng = 0,0,(0,0)#initialize number of x,y and gridpoints
        #将anchors大小缩放到grid尺度
        self.anchor_vec = self.anchors/self.stride
        #batch_size, na,grid_h,grid_w,wh
        #值为1的维度对应的值是不固定的,后续操作可根据broadcast广播机制自动扩充
        self.anchor_wh = self.anchor_vec.view(1,self.na,1,1,2)
        self.grid = None
    
    def create_grids(self,ng=(13,13),device="cpu"):
        '''
        更新grids信息并生成新的grids参数
        :param ng: 特征图大小
        :param device: 
        '''
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng,dtype=torch.float)
        
        #build xy offsets 构建每个cell处的anchor的xy偏移量(在feature map上的)
        if not self.training: #训练模式不需要回归到最终预测boxes
            yv,xv = torch.meshgrid([torch.arange(self.ny,device=device),
                                   torch.arange(self.nx,device=device)])
            #batch_size, na, grid_h, grid_w,wh
            self.grid = torch.stack((xv,yv),2).view((1,1,self.ny,self.nx,2)).float()
        
        if self.anchor_vec.device !=device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh=self.anchor_wh.to(device)
        
        def forward(self,p):
            bs,_,ny,nx=p.shape#batch_size, predict_param(255),gridy(13),gridx(13)
            if (self.nx,self.ny)!=(nx,ny) or self.grid is None:
                self.create_grids((nx,ny),p.device)
            
            #view:(batch_size,255,13,13)->(batch_size,3,85,13,13)
            #permute:(batch_size,3,85,13,13)->(batch_size,3,13,13,85)
            #[bs,anchors,grid,xywh+obj+classes]
            p=p.view(bs,self.na,self.no,self.ny,self.nx).permute(0,1,3,4,2).contiguous()#output prediction
        	
            if self.training:
                return p
            
            else:#inference
                #[bs,anchor,grid,xywh+obj+classes]
                io=p.clone()#inference output
                io[...,:2]=torch.sigmoid(io[...,:2])+self.grid#xy计算在feature map上的xy坐标
                io[...,:2:4]=torch.exp(io[...,2:4])*self.anchor_wh#wh yolo method 计算在feature map上的wh
                io[...,:4]*=self.stride#换算映射回原图尺度
                torch.sigmoid_(io[...,4:])
                return io.view(bs,-1,self.no), p#view [1,3,13,13,85] as [1,507,85]
                
        
```



### some tricks

- Bounding Box Prediction

  bbox预测手段是V3论文中的一个亮点. 在V2中, 借鉴faster R-CNN的RPN中的anchor机制. 但其存在一个问题, v2的anchor机制线性回归不稳定(因为回归的offset可以使用box偏移到图片的任何地方), 所以V3直接预测**相对位置**. 预测出bbox中心点相对网格单元左上角的相对坐标.

  ![image-20211201171949765](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\yolov3_bboxPrediction.png)

V3中对prior有明确解释: 选用bbox priors的k=9, 对于tiny-yoyo, k=6. priors都是在数据集上聚类得来, 有确当得数值

`[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198], [373,326]`

每个anchor prior是两个数字组成, 分别代表高宽.

V3对bbox进行预测时, 采用`logistic regression`. 类似于RPN中得线性回归调整bbox. V3每次对bbox进行predict时, 输出和V2一样, 都是$(t_x,t_y,t_w,t_h,t_o)$, 然后通过figure2 中得公式计算出绝对得$(x,y,w,h)$

logistic回归用于对anchor包围得部分进行一个目标性评分(objectness score),即这块位置是目标得可能性有多大. 这一步是在predict之前进行的, 可以去掉不必要的anchor, 减少计算量.

原文描述

> if the bounding box prior is not the best but does overlap a ground truth object by more than some threshold we ignore the prediction, following [17]. We use the threshould of 0.5. Unlike [17] our system only assigns one bounding box prior for each ground truth object.

- 其他

> 1. 9个anchor会被三个输出张量平分的. 根据大中小三种size各自取自己的anchor
> 2. 每个输出y在每个自己的网格都会输出3个预测框, 通过输出的张量维度, 13x13x255. 其255表示(3*(5_80). 
> 3. 作者使用logistics回归来对每个anchor包围的内容进行一个目标性评分(objectness score), 根据目标性评分来选择anchor prior进行predict, 而不是所有anchor prior都会输出.

- loss: 除了w,h的损失函数依然采用总方误差之外, 其他部分的损失函数用的是二值交叉熵.

### hyperparameters for train

| name         | value  | descript                                             |
| ------------ | ------ | ---------------------------------------------------- |
| giou         | 3.54   | giou loss gain                                       |
| cls          | 37.4   | cls loss gain                                        |
| cls_pw       | 1.0    | cls BCELoss positive_weight                          |
| obj          | 64.3   | obj loss gain                                        |
| obj_pw       | 1.0    | obj BCELoss positive_weight                          |
| iou_t        | 0.2    | iou training threshold                               |
| lr0          | 0.001  | initial learning rate (SGD=5E-3)                     |
| lrf          | 0.01   | final OneCycleLR learning rate (lr0*lrf)             |
| momentum     | 0.937  | SGD momentum                                         |
| weight_decay | 0.0005 | optimizer weights decay                              |
| fl_gamma     | 0.0    | focal loss gamma (efficientDet default is gamma=1.5) |
| hsv_h        | 0.0138 | image HSV-Hue augmentation (fraction)                |
| hsv_s        | 0.678  | image HSV-Saturation augmentatioin (fraction)        |
| hsv_v        | 0.36   | image HSV_value augmentation (fraction)              |
| degress      | 0.     | image rotation (+/- deg)                             |
| translate    | 0.     | image translation (+/- fraction)                     |
| scale        | 0.     | image scale (+/- gain)                               |
| shear        | 0.     | image shear (+/- deg)                                |

## YOLOV3 SPP

![img](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\yolov3_spp.png)

与YOLO V3的区别在于, 将第一个预测特征图经过Convolution Set给中间拆开, 并插入SPP模块.

YOLO V3的SPP模块并不是SPPnet 的SPP结构(Spatial Pyramid Pooling), 有所借鉴但是不同. SPP模块很简单, 首先输入直接接到输出作为第一个分支, 第二个分支是池化核为5x5的最大池化, 第二个分支是池化核为9x9的最大池化, 第三个分支是池化核为13x13的最大池化, 注意步距都是为1, 意味着池化前进行padding填充, 最后池化后得到的特征图尺寸大小核深度不变. 

![请添加图片描述](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\yolov3_sppTrait.png)

SPP模块的输入为16x16x512, 其输出为16x16x2048, 即深度扩充了4倍.

为什么第一个预测特征层前接了SPP结构呢?在第二,三个预测特征层前接上SPP结构会怎么样呢?

从下图可见, YOLOV3_SPP1和YOLOV3_SPP3的性能相近. 实验结果发现, 当输入尺度小的时候YOLOV3_SPP1还会好一点, 但随着输入尺度的增大, YOLOV-SPP3的性能回略好一些.

![请添加图片描述](E:\kuisu\typora\深度学习资料\目标检测\YOLO网络讲解-Chinese.assets\YOLOV3_SPPNum.png)
