# 全面梳理：准确率,精确率,召回率,查准率,查全率,假阳性,真阳性,PRC,ROC,AUC,F1

二分类问题的结果有四种

![img](E:\kuisu\typora\机器学习资料\全面梳理-准确率-精确率-召回率-AUC.assets\二分类举证.jpg)

-  准确率 accuracy

  正确分类的样本/总样本: $(TP+TN)/ALL$

  在不平衡分类问题中难以准确度量: 如98%的正样本只需全部预测为正确即可获得98%准确率

- 精确率-[查准率]-\[precision]

  $TP/(TP+FP)$

  在预测为1的样本中, 实际为1的概率

  查准率在检索系统中: 检出的相关文献与检出的全部文件的百分比, 衡量检索的信噪比

- 召回率-查全率-recall

  $TP/(TP+FN)$

  查全率在检索系统中: 查出的相关文献与全部相关相关文献的百分率, 衡量检索的覆盖率

常见分析

- 在肿瘤判断和地震预测等场景: 要求模型有更高的召回率(recall)
- 在垃圾邮件判断场景: 要求模型有更高的精确率(precision), 放入回收站里的可都确定是垃圾.

## ROC

常用来评价一个二分类器的优劣

![img](E:\kuisu\typora\机器学习资料\全面梳理-准确率-精确率-召回率-AUC.assets\ROC.jpg)

ROC曲线

- 横坐标

  false position rate(FPR): $FP/(FP+TN)$

  假阳率, 即实际无病, 但根据筛查被判断为有病的百分比.

- 纵坐标

  ture positive rate (TPR): $TP/(TP+FN)$

  真阳性率, 即实际有病, 根据筛查被判断有病的百分比.

  实际为1的样本中, 预测为1的概率, 基础为召回率-查全率-recall

考虑ROC曲线中的四个点和一条线

- 第一个点(0,1): 即FPR=0, TPR=1, 这意味着无病的都没有误判, 有病的全部都检测到, 这就是一个完美的分类器.
- 第二个点(1,0): 即FPR=1, TPR=0, 这是一个最糟糕的分类器, 因为他成功笔避开了所有的正确答案
- 第三个点(0,0), 即FPR＝TPR=0, 即没病的没有被误判, 但有病的全都没有检测到.即全部选0
- 第四个点(1,1), 与第三个点类似, 分类器实际所有的样本都为1

经过上诉分析, ROC曲线越接近左上角, 该分类器性能越好.

## 绘制ROC曲线

分类器有概率输出, 50%常被作为阈值点, 但基于不同的场景, 可以通过控制概率输出的阈值来改变预测的标签, 这样的不同阈值会得到不同的FPR和TPR

从0%-100%之间选取任意细度的阈值分别获得FPR和TPR，对应在图中，得到的ROC曲线，阈值的细度控制了曲线的阶梯程度或平滑程度。

ROC曲线有个很好的特性: 当测试集中的正负样本的分布变化的时候, ROC曲线能够保持不变. 

![img](E:\kuisu\typora\机器学习资料\全面梳理-准确率-精确率-召回率-AUC.assets\ROC-precisionRecall.jpg)

## AUC

AUC (Area under curve) 被定义为ROC曲线下的面积, 完全随机的二分类其的AUC为0.5, 虽然在不同的阈值下有不同的FPR和TPR, 但相对面积更大, 更靠近左上角的曲线代表着一个更加稳健的二分类器.

同时针对每一个ROC曲线, 又能找到一个最佳的概率切分点,使得自己预测的指标达到最佳水平.

## PR曲线

![img](E:\kuisu\typora\机器学习资料\全面梳理-准确率-精确率-召回率-AUC.assets\TP-FP-TN-FN.jpg)

我们希望检测的结果P越高越好, R也越高越好, 但实际这两者在某些情况下是矛盾的.

例如, 假设我们的数据集中共有5个待检测的物体, 我们的模型给出了10个候选框, 我们将按照模型给出的置信度进由高到低对候选框进行排序.

对于模型评价指标, VOC里面是叫AP, 而COCO数据集里较mAP, 主要是因为COCO数据集里mAP计算是针对10个IOU阈值下的AP值取平均(这个IOU阈值是np.linspace(0,5, 0.95, 10))

计算AP, 首先需要绘制P-R曲线, 也就是准确率-召回率曲线. 

模型的evaluation过程就是: 网络模型输出-->NMS-->AP计算

这里以测试集为一张图片为例, 假设共有(1,2,3)三种目标类别, 经过NMS后该图片中保留的bounding boxes及其对应的confidence score, prediction label如下表

![在这里插入图片描述](E:\kuisu\typora\机器学习资料\全面梳理-准确率-精确率-召回率-AUC.assets\AP-1.jpg)

至于上述的bounding boxes是TP还是FP, 就需要计算其ground truth的IOU值是否大于iou_thres来判断, iou_thres的取值不同即为不同的AP类型, 如iou_thres=0.5对应AP50.

判断bounding boxes是TP还是FP的过程如下

对于每个预测的bounding box, 如上表id=1的box, 器pred_label为1, 则计算其与该图片中所有类别为1的ground truth box的IOU值, 取其中最大的IOU值iou_max对应的ground truth box作为该预测box对应的ground truth box, 如果iou_max>iou_thres, 则该预测box为TP, 否则为FP.

 上表中的各个预测box的tp_label如下表所示, TP为1, FP为0

![在这里插入图片描述](E:\kuisu\typora\机器学习资料\全面梳理-准确率-精确率-召回率-AUC.assets\AP-2.jpg)

对每个类别需要单独计算AP, 最后所欲类别取平均. 

下面以类别1为例, 假设测试集图片中共有3个类别1的标注框(ground truth boxes), 显然上述预测结果没有将全部真值找回.

绘制P-R曲线, 结果如下表:

![在这里插入图片描述](E:\kuisu\typora\机器学习资料\全面梳理-准确率-精确率-召回率-AUC.assets\P-R curve.jpg)

P-R曲线如下图(准确率两端补0, 召回率两端分别补0和1)

> P是准确率，Ｒ是召回率。逐渐降低置信度conf_score的阈值，比如一开始置信度阈值是0.6，那只有id5是预测出来的，那么类别１的准确率和召回率就是１和１/3，然后降低置信度阈值为０.5，那么id5和id10 是预测出来的，那么类别１的准确率和召回率就是1/2和1/3，以此类推

![在这里插入图片描述](E:\kuisu\typora\机器学习资料\全面梳理-准确率-精确率-召回率-AUC.assets\P-R curve-1)

## AP(Average Precision)

顾名思义AP就是平均精确度, 简单来说就是PR曲线上的Precision值求均值. 杜宇pr曲线来说, 我们使用积分来进行计算.
$$
A P=\int_{0}^{1} p(r) d r
$$
实际应用中, 我们并不直接对该PR曲线进行计算, 而是对PR曲线进行平滑处理. 即对PR曲线上的每个点,  Precision的值取该点右侧的最大的Precision的值.

![preview](E:\kuisu\typora\机器学习资料\全面梳理-准确率-精确率-召回率-AUC.assets\PR curve 平滑处理.jpg)

用公式来描述就是$P_{\text {smooth }}(r)=\max _{r^{\prime}>=r} P\left(r^{\prime}\right)$​  。用该公式进行平滑后再用上述公式计算AP的值。

### Interplolated AP(Pascal Voc 2008的AP计算方式)

Pascal VOC 2008中设置IoU的阈值为0.5, 如果一个目标被重复检测, 则置信度最高的为正样本, 另一个为负样本. 在平滑处理的PR曲线上,取横轴0-1的10等分点的Precision的值, 计算其平均值为最终的AP的值
$$
AP= \frac{1}{11} \sum_0^1 P_{smooth}(i)
$$
![img](E:\kuisu\typora\机器学习资料\全面梳理-准确率-精确率-召回率-AUC.assets\PascaVoc2008.jpg)
$$
AP=\frac{1}{11}(5×0.1+4×0.57+2×0.5)=0.753
$$

### COCO mAP

最小的目标检测相关论文都使用coco数据集来展示自己模型效果. 对于CoCO数据集来说, 使用的也是Interplolated AP的计算方式. 与VOC 2008不同的是, 为了提高精度, 在PR曲线上采样了100个点进行计算. 而且IOU的阈值从固定的0.5调整为0.5~0.95的区间上每隔0.5计算一次AP的值, 所有结果的平均值为最终的结果.

## 参考

- [全面梳理](https://zhuanlan.zhihu.com/p/34079183)
- [mAP](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)

