## Faster RCNN

## 简介

Faster RCNN其实可以分为4个主要内容：

1. Conv layers: 作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv+relu+pooling层提取image的feature maps。该feature maps被共享用于后续RPN层和全连接层。
2. Region Proposal Networks: RPN网络用于生成region proposals。该层通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的proposals。
3. Roi Pooling: 该层收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。
4. Classification: 利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置。

所以本文以上述4个内容作为切入点介绍Faster R-CNN网络。

下图展示了python版本中的VGG16模型中的faster_rcnn_test.pt的网络结构，可以清晰的看到该网络对于一副任意大小PxQ的图像：

- 首先缩放至固定大小MxN，然后将MxN图像送入网络；
- 而Conv layers中包含了13个conv层+13个relu层+4个pooling层；
- RPN网络首先经过3x3卷积，再分别生成positive anchors和对应bounding box regression偏移量，然后计算出proposals；
- 而Roi Pooling层则利用proposals从feature maps中提取proposal feature送入后续全连接和softmax网络作classification（即分类proposal到底是什么object）。

![preview](E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\Faster RCNN structure.jpg)

## 1. Conv layers

Conv layers包含了conv，pooling，relu三种层。以python版本中的VGG16模型中的faster_rcnn_test.pt的网络结构为例，如图2，Conv layers部分共有13个conv层，13个relu层，4个pooling层。这里有一个非常容易被忽略但是又无比重要的信息，在Conv layers中：

1. 所有的conv层都是：kernel_size=3，pad=1，stride=1
2. 所有的pooling层都是：kernel_size=2，pad=0，stride=2

在Faster RCNN Conv layers中对所有的卷积都做了扩边处理（ pad=1，即填充一圈0），导致原图变为 (M+2)x(N+2)大小，再做3x3卷积后输出MxN 。正是这种设置，导致Conv layers中的conv层不改变输入和输出矩阵大小。如图3：

![img](E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\CNN_pooling.jpg)

类似的是，Conv layers中的pooling层kernel_size=2，stride=2。这样每个经过pooling层的MxN矩阵，都会变为(M/2)x(N/2)大小。综上所述，在整个Conv layers中，conv和relu层不改变输入输出大小，只有pooling层使输出长宽都变为输入的1/2。

那么，一个MxN大小的矩阵经过Conv layers固定变为(M/16)x(N/16)！这样Conv layers生成的feature map中都可以和原图对应起来。

## 2 Region Proposal Networks(RPN)

这部分其实可以看成是One-Stage检测器的检测输出部分. 实际上对于只检测一类目标来说, 可以直接拿来用. RPN在Faster RCNN中的作用是结合先验的Anchor, 将背景和前景区分开来(二分类), 这样的话大量的先验anchor就可以被筛选出来, 并作些回归(使得anchor更接近于真实目标).

经典的检测方法生成检测框都非常耗时，如OpenCV adaboost使用滑动窗口+图像金字塔生成检测框；或如R-CNN使用SS(Selective Search)方法生成检测框。而Faster RCNN则抛弃了传统的滑动窗口和SS方法，直接使用RPN生成检测框，这也是Faster R-CNN的巨大优势，能极大提升检测框的生成速度。

![img](E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\RPN.jpg)

上图4展示了RPN网络的具体结构。可以看到RPN网络实际分为2条线，上面一条通过softmax分类anchors获得positive和negative分类，下面一条用于计算对于anchors的bounding box regression偏移量，以获得精确的proposal。而最后的Proposal层则负责综合positive anchors和对应bounding box regression偏移量获取proposals，同时剔除太小和超出边界的proposals。其实整个网络到了Proposal Layer这里，就完成了相当于目标定位的功能。

![image-20211011121951595](E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\RPN visual.png)

## 2.1 anchors

官方pytorch torchvision里的Faster RCNN代码中, 输入图像尺度为768x1344, 5个feature map分别经过了stride=(4,8, 16, 32, 64), 得到了5个大小为(192x336, 96x168, 48x84, 24x42, 12x21)的feature. 

代码中预定义了5个尺度(32, 64, 128, 256, 512), 3种aspect_ratio (0.5, 1.0, 2.0)的Anchor. 这样的话, 我们可以得到5组base_anchor, 每一组包含3个面积相同, 宽高比不同的以原点为中心的基础锚框.

![image-20211012185343873](E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\anchors.png)

然后将这些base_anchor撒到对应的feature map上(起点是[0,0]). 这样的话, 共有:

$(192*336*3)+(96*168*3)+(48*84*3)+(24*42*3)+(12*24)=257796$个anchor.

提到RPN网络，就不能不说anchors。所谓anchors，实际上就是一组由rpn/generate_anchors.py生成的矩形。直接运行的generate_anchors.py可以得到以下输出：

```text
[[ -84.  -40.   99.   55.]
 [-176.  -88.  191.  103.]
 [-360. -184.  375.  199.]
 [ -56.  -56.   71.   71.]
 [-120. -120.  135.  135.]
 [-248. -248.  263.  263.]
 [ -36.  -80.   51.   95.]
 [ -80. -168.   95.  183.]
 [-168. -344.  183.  359.]]
```

其中每行的4个值 ![[公式]](https://www.zhihu.com/equation?tex=%28x_1%2C+y_1%2C+x_2%2C+y_2%29) 表矩形左上和右下角点坐标。9个矩形共有3种形状，长宽比为大约为 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bwidth%3Aheight%7D%5Cin%5C%7B1%3A1%2C+1%3A2%2C+2%3A1%5C%7D) 三种，如图6。实际上通过anchors就引入了检测中常用到的多尺度方法。

![img](E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\anchor1.jpg)

注：关于上面的anchors size，其实是根据检测图像设置的。在python demo中，会把任意大小的输入图像reshape成800x600（即图2中的M=800，N=600）。再回头来看anchors的大小，anchors中长宽1:2中最大为352x704，长宽2:1中最大736x384，基本是cover了800x600的各个尺度和形状。

那么这9个anchors是做什么的呢？借用Faster RCNN论文中的原图，遍历Conv layers计算获得的feature maps，为每一个点都配备这9种anchors作为初始的检测框。这样做获得检测框很不准确，不用担心，后面还有2次bounding box regression可以修正检测框位置。

![img](E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\anchor-2.jpg)

解释一下上面这张图的数字。

1. 在原文中使用的是ZF model中，其Conv Layers中最后的conv5层num_output=256，对应生成256张特征图，所以相当于feature map每个点都是256-dimensions
2. 在conv5之后，做了rpn_conv/3x3卷积且num_output=256，相当于每个点又融合了周围3x3的空间信息（猜测这样做也许更鲁棒？反正我没测试），同时256-d不变（如图中的红框）
3. 假设在conv5 feature map中每个点上有k个anchor（默认k=9），而每个anhcor要分positive和negative，所以每个点由256d feature转化为cls=2•k scores；而每个anchor都有(x, y, w, h)对应4个偏移量，所以reg=4•k coordinates
4. 补充一点，全部anchors拿去训练太多了，训练程序会在合适的anchors中**随机**选取128个postive anchors+128个negative anchors进行训练（什么是合适的anchors下文5.1有解释）

注意，在本文讲解中使用的VGG conv5 num_output=512，所以是512d，其他类似。

**其实RPN最终就是在原图尺度上，设置了密密麻麻的候选Anchor。然后用cnn去判断哪些Anchor是里面有目标的positive anchor，哪些是没目标的negative anchor。所以，仅仅是个二分类而已！**

### anchor个数计算实例

那么Anchor一共有多少个？原图800x600，VGG下采样16倍，feature map每个点设置9个Anchor，所以：
$$
ceil(800/16)×ceil(600/16)×9=50×38×9=17100
$$
其中ceil()表示向上取整，是因为VGG输出的feature map size= 50*38。

![img](E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\v2-4b15828dfee19be726835b671748cc4d_b.jpg)

### anchorGenerator实现

```python

class AnchorsGenerator(nn.Module):
    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    """
    anchors生成器
    Module that generates anchors for a set of feature maps and image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Arguments:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()

        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}#定义空字典, 存储anchor

    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device=torch.device("cpu")):
        # type: (List[int], List[float], torch.dtype, torch.device) -> Tensor
        """
        compute anchor sizes
        Arguments:
            scales: sqrt(anchor_area)
            aspect_ratios: h/w ratios
            dtype: float32
            device: cpu/gpu
        """
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios

        # [r1, r2, r3]' * [s1, s2, s3]
        # number of elements is len(ratios)*len(scales)
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        # left-top, right-bottom coordinate relative to anchor center(0, 0)
        # 生成的anchors模板都是以（0, 0）为中心的, shape [len(ratios)*len(scales), 4]
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

        return base_anchors.round()  # round 四舍五入

    def set_cell_anchors(self, dtype, device):
        # type: (torch.dtype, torch.device) -> None
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            # suppose that all anchors have the same device
            # which is a valid assumption in the current state of the codebase
            if cell_anchors[0].device == device:
                return

        # 根据提供的sizes和aspect_ratios生成anchors模板
        # anchors模板都是以(0, 0)为中心的anchor
        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        # 计算每个预测特征层上每个滑动窗口的预测目标数
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """
        anchors position in grid coordinate axis map into origin image
        计算预测特征图对应原始图像上的所有anchors的坐标
        Args:
            grid_sizes: 预测特征矩阵的height和width
            strides: 预测特征矩阵上一步对应原始图像上的步距
        """
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        # 遍历每个预测特征层的grid_size，strides和cell_anchors
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            # shape: [grid_width] 对应原图上的x坐标(列)
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            # shape: [grid_height] 对应原图上的y坐标(行)
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height

            # 计算预测特征矩阵上每个点对应原图上的坐标(anchors模板的坐标偏移量)
            # torch.meshgrid函数分别传入行坐标和列坐标，生成网格行坐标矩阵和网格列坐标矩阵
            # shape: [grid_height, grid_width]
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            # 计算anchors坐标(xmin, ymin, xmax, ymax)在原图上的坐标偏移量
            # shape: [grid_width*grid_height, 4]
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            # 将anchors模板与原图上的坐标偏移量相加得到原图上所有anchors的坐标信息(shape不同时会使用广播机制)
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))

        return anchors  # List[Tensor(all_num_anchors, 4)]

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """将计算得到的所有anchors信息进行缓存"""
        key = str(grid_sizes) + str(strides)
        # self._cache是字典类型
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        # type: (ImageList, List[Tensor]) -> List[Tensor]
        # 获取每个预测特征层的尺寸(height, width)
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])

        # 获取输入图像的height和width
        image_size = image_list.tensors.shape[-2:]

        # 获取变量类型和设备类型
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        # one step in feature map equate n pixel stride in origin image
        # 计算特征层上的一步等于原始图像上的步长
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]

        # 根据提供的sizes和aspect_ratios生成anchors模板
        self.set_cell_anchors(dtype, device)

        # 计算/读取所有anchors的坐标信息（这里的anchors信息是映射到原图上的所有anchors信息，不是anchors模板）
        # 得到的是一个list列表，对应每张预测特征图映射回原图的anchors坐标信息
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])
        # 遍历一个batch中的每张图像
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            # 遍历每张预测特征图映射回原图的anchors坐标信息
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        # 将每一张图像的所有预测特征层的anchors坐标信息拼接在一起
        # anchors是个list，每个元素为一张图像的所有anchors信息
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        # Clear the cache in case that memory leaks.
        self._cache.clear()
        return anchors
```



## 2.3 softmax判定positive与negative

## FPN

特征提取用, 采用FPN网络(自下而上, 自上而下, 横向连接, 卷积融合)

![img](E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\FPN.png)

## RPN Head

![img](E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\RPNHead.png)

将特征提取网络获得的feature map先经过1x1卷积, 然后分别用两个3x3的卷积进行分类和回归操作. 以大小为256x48x84的feature map为例, 经过1x1卷积,不改变特征图尺寸, 假定feature map上宽高平面上每个点有3个anchor(宽高比分别是0.5, 1.0, 2.0), 那个分类支路上的3x3卷积输出维度是3x48x84, 而回归支路上的3x3卷积输出维度是12x48x84.

有多少Anchor, 就有多少分类回归结果, 最终多个尺度( FPN 5 个尺度)cancat后的分类维度为257796x1, 回归维度为257796x4. 值得注意的是, 这里的回归结果是基于编码后的Anchor的偏移量. 这就与Faster RCNN中的encode和decode有关, 区别于SSD和YOLO V3两个算法检测的编码方式

记$x,y,w,h$时检测框的中心点坐标和宽高, $x,x_a,x^*$分别代表检测框, Anchor和GT的对象坐标, $t_x$是检测偏移量

```python
proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
```

![image-20211012190937065](E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\decode.png)

### decode实现

```python
#解码
    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes (bbox regression parameters)
            boxes (Tensor): reference boxes (anchors/proposals)
        """
        boxes = boxes.to(rel_codes.dtype)

        # xmin, ymin, xmax, ymax
        widths = boxes[:, 2] - boxes[:, 0]   # anchor/proposal宽度
        heights = boxes[:, 3] - boxes[:, 1]  # anchor/proposal高度
        ctr_x = boxes[:, 0] + 0.5 * widths   # anchor/proposal中心x坐标
        ctr_y = boxes[:, 1] + 0.5 * heights  # anchor/proposal中心y坐标

        wx, wy, ww, wh = self.weights  # RPN中为[1,1,1,1], fastrcnn中为[10,10,5,5]
        dx = rel_codes[:, 0::4] / wx   # 预测anchors/proposals的中心坐标x回归参数
        dy = rel_codes[:, 1::4] / wy   # 预测anchors/proposals的中心坐标y回归参数
        dw = rel_codes[:, 2::4] / ww   # 预测anchors/proposals的宽度回归参数
        dh = rel_codes[:, 3::4] / wh   # 预测anchors/proposals的高度回归参数

        # limit max value, prevent sending too large values into torch.exp()
        # self.bbox_xform_clip=math.log(1000. / 16)   4.135
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # xmin
        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymin
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        # xmax
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymax
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h

        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes
```



同理, 在计算loss时, 我们需要将GT进行编码

```python
regression_targets=self.box_coder.encode(matched_gt_boxes,anchors)
```

![image-20211012191215065](E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\encode.png)

### encode实现

```python
#编码
@torch.jit._script_if_tracing
def encode_boxes(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """

    Encode a set of proposals with respect to some
    reference boxes

    Arguments:
        reference_boxes (Tensor): reference boxes(gt)
        proposals (Tensor): boxes to be encoded(anchors)
        weights:
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    # unsqueeze()
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    # parse widths and heights
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    # parse coordinate of center point
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets

```

 

然后, 将Anchor + BBox\_reg转换成Proposal, 注意每个proposal的物理意义时图片上的$x\_min,y\_min,x\_max,y\_max$​

### 筛选Proposal

首先每个检测尺度上筛选出前$min(pre\_nms\_top\_n, num\_anchors)$个anchor, (上面257796个proposal就会筛选出4756个), 用scale\_level来标记每个Proposal的尺度等级, 值域集合为{0, 1, 2, 3, 4};

随后对这些筛选出来的Proposal, 按照clip, remove\_small\_boxes, 每个尺度上分别做nms(具体实现是把所有的proposal+(scale\_level*Proposal.max())来加速操作)至多保留post\_nms\_top\_n个Proposal

```python
boxes, scores = self.filter_proposals(proposals, objectness, images.image_size,num_anchors_per_level)
```
### 代码实现

```python
   def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        """
        筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        Args:
            proposals: 预测的bbox坐标
            objectness: 预测的目标概率
            image_shapes: batch中每张图片的size信息
            num_anchors_per_level: 每个预测特征层上预测anchors的数目

        Returns:

        """
        num_images = proposals.shape[0]
        device = proposals.device

        # do not backprop throught objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        # Returns a tensor of size size filled with fill_value
        # levels负责记录分隔不同预测特征层上的anchors索引信息
        levels = [torch.full((n, ), idx, dtype=torch.int64, device=device)
                  for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0)

        # Expand this tensor to the same size as objectness
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        # 获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]  # [batch_size, 1]

        # 根据每个预测特征层预测概率排前pre_nms_top_n的anchors索引值获取相应概率信息
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        # 预测概率排前pre_nms_top_n的anchors索引值获取相应bbox坐标信息
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        # 遍历每张图像的相关预测信息
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            # 调整预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # 返回boxes满足宽，高都大于min_size的索引
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # 移除小概率boxes，参考下面这个链接
            # https://github.com/pytorch/vision/pull/3205
            keep = torch.where(torch.ge(scores, self.score_thresh))[0]  # ge: >=
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores
```



## RoI Pooling

有了筛选出来的Proposal, 就可以将其映射到某个feature map上, 然后利用RoI Pooling提取每个Proposal的特征供后续的RCNN细致的分类和回归了.

```python
box_feature = self.box_roi_pool(features, proposals, image_shapes)
```

代码中只选择了其中4个尺度来进行操作(最小的那个feature map 没用到)

因为是多尺度特征, 把每个proposal映射哪个feature map上是个问题,代码用LevelMapper来操作
$$
k=[k_0+log 2 (\sqrt{wh}/224)]
$$
其中k_0是一个面积为224x224的proposal应该处于的feature feature map level.其他尺度的proposal按照上面的公式安排feature map level

代码选择256x48x48这个尺度的feature map为k_0, 对应224x224这个大小范围的Proposal. 

随后使用roi_align提取每个proposal的特征

```python
result_idx_in_level = roi_align(per_level_feature, rois_per_level,
                               output_size=self.output_size,
                               spatial_scale=scale,
                               sampling_ratio=self.sampling_ratio)
```

这样的话, 我们就可以获得post_nms_top_n个256x7x7维度的前景框的特征. 最后的flatten特征维度接上两个全连接层获得post_nms_top_n 个1024维度的特征.

### 损失计算

- smooth L1 loss

  对于边框的预测是一个回归问题, 通常可以选择平方损失函数(L2损失), 但这个损失对于比较大的误差的惩罚项很高. 

  我们可以采用稍微缓和一点的绝对损失函数(L1损失) $f(x)=|x|$, 但这个函数再0点处导数不存在, 因此可以能回影响收敛.

  通常的解决办法是用分段函数, 在0点附近使用屏方函数使得它更平滑. 其通过一个参数$\delta$​来控制平滑区域. 

  ![img](E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01yX2hlYWx0aA==,size_16,color_FFFFFF,t_70)

  

- Faster RCNN损失函数

  Faster RCNN的损失主要分为RPN损失和Faset RCNN损失, 且两部分损失都包括分类损失和回归损失.

![img](E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\20181108105638545.png)

- RPN分类损失

  RPN网络产生的anchor只分为前景和背景, 前景的标签为1, 背景的标签为0. 在训练RPN的过程中, 回选择256个anchor.

  ![img](E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\20181212133229147.png)

### 损失计算实现

```python
def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits : 预测类别概率信息，shape=[num_anchors, num_classes]
        box_regression : 预测边目标界框回归信息
        labels : 真实类别信息
        regression_targets : 真实目标边界框信息

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # 计算类别损失信息
    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    # 返回标签类别大于0的索引
    # sampled_pos_inds_subset = torch.nonzero(torch.gt(labels, 0)).squeeze(1)
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]

    # 返回标签类别大于0位置的类别信息
    labels_pos = labels[sampled_pos_inds_subset]

    # shape=[num_proposal, num_classes]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    # 计算边界框损失信息
    box_loss = det_utils.smooth_l1_loss(
        # 获取指定索引proposal的指定类别box信息
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    ) / labels.numel()

    return classification_loss, box_loss
```

## 其他

### smooth_l1_loss

```python
torch.nn.functional.smooth_l1_loss(input,target,size_average=None,reduce=None, reduction='mean')
```

![在这里插入图片描述](E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\loss.png)

### VGG16-微调

<img src="E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\vgg16_struct.jpg" alt="img" style="zoom:50%;" />

```python
backbone = torch.nn.Sequential(*list(vgg_feature._modules.values()))[:-1]
```

`vgg.features`是取出vgg16网络中的features大层。其中vgg网络可以分为3大层，一层是（features），一层是（avgpool），最后一层是（classifier）

`vgg.features._modules`: 将取出来的网络转为字典显示

`vgg.features._modules.values()`: 及那个字典对应键值都取出,即将各个小层的网络参数取出

`list(vgg.features._modules.values())[:1]`: 将这些网络层参数强制转为list类型

### FPN的构建

#### 主体结构

```python

def resnet50_fpn_backbone(pretrain_path='',
                          norm_layer=FrozenBatchNorm2d,
                          trainable_layer=3,
                          returned_layers=None,
                          extra_blocks=None):
    """
    搭建resnet50_fpn--backbone
    :param pretrain_path: resnet50的预训练权重, 如果不适用, 默认为空
    :param norm_layer: 官方默认的时ForzenBatchNoram2d, 即不会跟新参数的bn层(因为如果batch_size
        设置很小会导致效果更差, 如果GPU显存大可以设置很大的batch_size, 那么可以传入正常的BatchNorm2d
    :param trainable_layer: 指定训练哪些层结构
    :param returned_layers: 指定哪些层的输出需要返回
    :param extra_blocks: 再输出的特征层基础上额外添加的层结构
    :return:
    """
    #1. 骨干网络
    resnet_backbone = ResNet(Bottleneck,[3,4,6,3],
                             include_top=False,
                             norm_layer=norm_layer)
    if isinstance(norm_layer,FrozenBatchNorm2d):
        overwrite_eps(resnet_backbone,0.0)

    if pretrain_path !="":
        assert os.path.exists(pretrain_path), "{} is not exist.".format(pretrain_path)
        #载入预训练权重
        print(resnet_backbone.load_state_dict(torch.load(pretrain_path),strict=False))

    #2. select layers that wont  be frozen
    assert 0 <= trainable_layer <=5
    layers_to_train = ['layer4','layer3','layer2','layer1','conv1'[:trainable_layer]]

    #如果要训练所有层的结构的话, 不要忘了conv1后还有一个bn1
    if trainable_layer == 5:
        layers_to_train.append('bn1')

    #3. freeze layers
    for name, parameter in resnet_backbone.named_parameters():
        #只训练不在layers_to_train列表中的层结构
        if all([not name.startswith(layer) for layer in layers_to_train]):#startswith非常重要
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1,2,3,4]

    #返回的特征层个数肯定大于0小于5
    assert min(returned_layers) >0 and max(returned_layers)<5

    #4. return_layers = {"layer1":"0", "layer2":"1","layer3":"2","layer4":"3"}
    return_layers = {f"layer{k}": str(v) for v,k in enumerate(returned_layers)}

    #5. 获取需要进行FPN拼接层中, 每一层的输出通道数:
    #in_channel 为layer4的输出特征矩阵channel=2048
    in_channels_stage2 = resnet_backbone.in_channel//8 #256

    #6. 记录resnet50提供给fpn的每个特征层channel
    in_channels_list = [in_channels_stage2*2**(i-1) for i in returned_layers]

    #7. 通过fpn后得到的每个特征层的channel
    out_channels = 256
    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list,out_channels,
                           extra_blocks=extra_blocks)
```

#### FPN的主体结构

```python
class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    :argument:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]):a dict containing the names of the
            modules for which the activations will be returned as the key of the dict,
            and the value of the dict is the name of the returned activation (which
            the user can specify).
        in_channels_list (List[int]): number of channels for each feature map that
            is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        extra_blocks: ExtraFPNBlock
    Atrributes:
        out_channels (int): the number of channels in the FPN
    """
    def __init__(self, backbone,return_layers,in_channels_list,out_channels,extra_blocks=None):
        super(BackboneWithFPN, self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
		#1. 获取中间层
        self.body = IntermediateLayerGetter(backbone,return_layers=return_layers)
        #2. 获取fpn层
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list,
                                         out_channels=out_channels,
                                         extra_blocks=extra_blocks)
        self.out_channels = out_channels

    def forward(self,x):
        x = self.body(x)
        x= self.fpn(x)
        return x
```

#### 获取中间层

```python

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model. It has
    a strong assumption that the modules have been registered into the model
    into the model in the same order as they are used.

    This means that one should not reuse the same nn.Module twice in the
    forward if you want this to work.
    Additionally, it is only anble to query submodules that are directly
    assigned to the model. So if model is passed, model.feature1 can be
    returned, but not model.feature1.layers.
    :argument:
        model (nn.Module): model on which we will extract the features
        :return_layers Dict([name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    __annotations__ = {
        "return_layers": Dict[str,str]
    }
    def __init__(self,model,return_layers):
        #2. 保证所有的return_layers的keys都在model中
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {str(k):str(v) for k,v in return_layers.items()}
        layers = OrderedDict()

        #3. 将要从model中获取信息的最后一层之前的模块全部复制下来
        #即,再resnet50中, 只保留layer4及其之前的结构,舍去之后不用的结构
        for name, module in model.named_children():
            layers[name] =module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)#将所需的网络层通过继承的方式保存下来
        self.return_layers = orig_return_layers


    def forward(self,x):
        out = OrderedDict()
        #4. 依次遍历模型的所有子模块, 并进行正向传播,
        #收集layer1, layer2, layer3, layer4的输出
        #将所需的值以k,v的形式保存到out中, 这里的self.items()就是layers
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
```

#### 构建FPN层

```python

class FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    'feature pyramid network for object detection" <https://arxiv.org/abs/1612.03144>`_.

    The feature maps are currently supposed to be in increasing depth order.
    The input to the model is expected to be an OrderedDict[Tensor], containing the
    features maps on top of which the FPN will be added
    :argument
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels(int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will be
            performed. It is expected to take the fpn features, the original features
            and the names of the original features as input, and returns a new list of
            feature maps and their corresponding names
    """
    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        super(FeaturePyramidNetwork, self).__init__()
        #用来调用resnet特征矩阵(layer 1,2,3,4)的channel (kernel_size=1)
        self.inner_blocks = nn.ModuleList()
        #对调整后的特征矩阵使用3x3的卷积核来得到对应的预测特征矩阵
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3,padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        #initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight,a=1)
                nn.init.constant_(m.bias,0)
        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self,x,idx):
        #type: (Tensor,int)->Tensor
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        :param x:
        :param idx:
        :return:
        """
        num_blocks = len(self.inner_blocks)
        if idx<0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i+=1
        return out

    def get_result_from_layer_blocks(self,x,idx):
        """
        This is equivalent to self.layer_blocks[idx](x)
        but torchscript doesn't support this yet
        :param x:
        :param idx:
        :return:
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i=0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self,x):
        #type: (Dict[str, Tensor])->Dict[str,Tensor]
        """
        Computes the FPN for a set of feature maps.
        :param x: (OrderDict[Tensor]): feature maps for each feature level.
        :return: (OrderDict[Tensor]): feature maps after FPN layer. They
            are ordered from highest resolution first.
        """
        #unpack OrderedDict into two lists for easier handing
        names = list(x.keys())
        x = list(x.values())

        #将resnet layer4的channel调整到指定的out_channels
        #last_inner = self.inner_blocks[-1](x[-1])
        last_inner = self.get_result_from_inner_blocks(x[-1],-1)#获取最后一层

        #result中保存这每个预测特征层
        results = []

        #将layer4 调整channel后的特征矩阵, 通过3x3卷积后得到对应的预测特征矩阵
        # results.append(self.layer_blocks[-1](last_inner))
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        #一个自底向上的线路, 一个自顶向下的线路, 横向连接(latteral connection)
        for idx in range(len(x)-2,-1,-1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx],idx)#conv3x3, 将通道数变为256
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape,mode='nearest')#将最后一层进行上采样
            last_inner = inner_lateral+inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner,idx))#最高分辨率的在前面

        #在layer4对应的预测特征层基础上生成预测特征矩阵5
        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results,x, names)

        #make it back an OrderedDict
        out = OrderedDict([(k,v) for k,v in zip(names,results)])

        return out
```



### trian实验记录

- 从头开始训练

  ```python
  #最开始的训练损失情况
  lr: 0.000010  loss: 4.2163 (4.2163)  loss_classifier: 3.0421 (3.0421)  loss_box_reg: 0.1964 (0.1964)  loss_objectness: 0.6930 (0.6930)  loss_rpn_box_reg: 0.2849 (0.2849)
                          
  #训练5个epoch的损失情况
  lr: 0.005000  loss: 2.2868 (1.9905)  loss_classifier: 1.0894 (0.9944)  loss_box_reg: 0.6600 (0.5261)  loss_objectness: 0.2953 (0.3350)  loss_rpn_box_reg: 0.1471 (0.1350)
  ```

  ![image-20211018144716049](E:\kuisu\typora\深度学习资料\目标检测\Faster RCNN.assets\image-20211018144716049-16345396378451.png)

