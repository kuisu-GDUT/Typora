# Mask-RCNN官解析

## 模型定义(modeling)-关键部分

无论在训练文件还是测试文件中, 都是用了`build_detection_model(cfg)`函数来创建模型, 该函数可以通过配置文件组合出不同类型的模型, 为了了解模型的内部定义细节, 对`./maskrcnn_benchmark/modeling`下的文件进行分析

![image-20211108152541684](E:\kuisu\typora\深度学习资料\maskrcnn\maskrcnn_benchmark解析.assets\image-20211108152541684-16363563433071.png)

### detector

定义了模型入口

#### detectors.py文件解析

根据给定的配置信息实例化一个`class GeneralizedRCNN`的对象

```python
from .generalized_rcnn import GeneralizedRCNN

# 该函数是创建模型的入口函数，也是唯一的模型创建函数
_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}

def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]# 构建一个模型字典，虽然只有一对键值，但是方便后续的扩展
    return meta_arch(cfg)
	# 上面的语句等价于
    # return GeneralizedRCNN(cfg)
```

上面代码利用配置信息cfg实例化了一个`class GeneralizedRCNN`类, 该类定义在`./maskrcnn_benchmark/modeling/detector/generalized_rcnn.py`

#### generalized_rcnn.py

```python
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.#利用前面网络输出的features和proposal来计算detections/masks
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)#根据配置信息创建backbone网络
        self.rpn = build_rpn(cfg, self.backbone.out_channels)#根据配置信息创建rpn
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)#根据配置信息创建roi_heads

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:#如果roi_heads不为none的话, 就直接计算其输出的结果
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:#训练模式下, 输出其损失值
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result#不在训练模式下, 则输出模型的预测结果
```

可以看出, `MaskrcnnBenchmark`模型的创建主要依赖于三个函数, 即`build_backbone(cfg), build_rpn(cfg), build_roi_heads(cfg)`

### backbone[关于模型骨架的定义]

#### backbone.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict#有序字典

from torch import nn

#注册器, 用于管理module的注册,使得可以像使用字典一样使用module
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet

#创建resnet骨架网络, 根据配置信息会被后面的build_backbone()函数调用
@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)#resnet.py文件中的class Resnet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))#利用nn.Squential定义模型
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model

#创建具有fpn结构的骨干网络
@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    #1. 创建resnet网络
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    #创建FPN网络, (利用fpn.py文件下的class FPN)
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

#利用上述函数进行模型构建
def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
```

#### resnet.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
OR:
    model = ResNet(
        "StemWithGN",
        "BottleneckWithGN",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import DFConv2d
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.utils.registry import Registry


# ResNet stage specification
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Number of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

# -----------------------------------------------------------------------------
# Standard ResNet models
# -----------------------------------------------------------------------------
# ResNet-50 (including all stages)
ResNet50StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True))
)
# ResNet-50 up to stage 4 (excludes stage 5)
ResNet50StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, True))
)
# ResNet-101 (including all stages)
ResNet101StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True))
)
# ResNet-101 up to stage 4 (excludes stage 5)
ResNet101StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, True))
)
# ResNet-50-FPN (including all stages)
ResNet50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
)
# ResNet-101-FPN (including all stages)
ResNet101FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True))
)
# ResNet-152-FPN (including all stages)
ResNet152FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 8, True), (3, 36, True), (4, 3, True))
)

class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        #将配置文件中的字符串转为具体的实现, 下面分三个分别使用了对应的注册模块
        # Translate string names to implementations, stem的实现, 也就是resnet的第一阶段conv1
        # 1. cfg.MODEL.RESNETS.STEM_FUNC = 'StemWithFixedBatchNorm'
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        # 2. rennet conv2_x~conv5_x的实现
        # eg: cfg.MODEL.CONV_BODY= "R-50-FPN"
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        #3. residual transformation function
        # eg: cfg.MODEL.RESNETS.TRANS_FUNC="BottleneckWithFixedBatchNorm"
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]

        # Construct the stem module
        self.stem = stem_module(cfg)

        # Constuct the specified ResNet stages
        # 当num_gropus=1 时为resnet, >1时为resnext
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        #in_channels: 向后面的第二阶段输入时特征图的通道数, 也就是stem的输出通道数, 默认为64
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        stage2_bottleneck_channels = num_groups * width_per_group
        # 第二阶段的输出, resnet系列标准的模型, 可以从resnet第二阶段的输出通道判断后续的通道数
        # 默认为256, 则后续分别为512, 1024, 2048, 若为64, 则后续为128, 256, 512
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        # 创建一个空的stages列表和对应的特征图字典
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            # 计算每一个stage的输出通道数, 每经过一个stage, 通道数都会加倍
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            # 计算输入特征图的通道数
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index -1]
            # 该函数可以根据传入的参数创建对应stage的模块
            # 当获取到所有需要的参数以后, 调用本文件的-make_stage函数
            module = _make_stage(
                transformation_module,
                in_channels,
                bottleneck_channels,#压缩后的通道数
                out_channels,
                stage_spec.block_count,#当前stage的卷积层数量
                num_groups,#resnet时为1, resnext时为>1
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                # 当处于stage3~5时, 需要在开始的时候使用stride=2来downsize
                first_stride=int(stage_spec.index > 1) + 1,
                dcn_config={
                    "stage_with_dcn": stage_with_dcn,
                    "with_modulated_dcn": cfg.MODEL.RESNETS.WITH_MODULATED_DCN,
                    "deformable_groups": cfg.MODEL.RESNETS.DEFORMABLE_GROUPS,
                }
            )
            # 下一个stage的输入通道即为当前stage的输入通道数
            in_channels = out_channels
            # 当前stage模块添加到模型中
            self.add_module(name, module)
            # 将stage的名称添加到列表中
            self.stages.append(name)
            # 将stage的布尔值添加到字典中
            self.return_features[name] = stage_spec.return_features

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    # 将指定层的参数为: requires_grad = False
    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
                # 将m中的所有参数设置为不更新的状态
            for p in m.parameters():
                p.requires_grad = False

    #定义resnet前向传播过程
    def forward(self, x):
        outputs = []
        x = self.stem(x)#先经过stem (stage 1)
        #再一次计算stage2~5的结果
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                #将stage2~5的计算结果(特征图)以列表的形式保存
                outputs.append(x)
        #outputs为列表形式, 元素为各个stage的特征图, 正好作为FPN的输入.
        return outputs


class ResNetHead(nn.Module):
    def __init__(
        self,
        block_module,
        stages,
        num_groups=1,
        width_per_group=64,
        stride_in_1x1=True,
        stride_init=None,
        res2_out_channels=256,
        dilation=1,
        dcn_config={}
    ):
        super(ResNetHead, self).__init__()

        #获取不同stage的对应的通道数, 其相对于stage2的倍数
        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor

        block_module = _TRANSFORMATION_MODULES[block_module]

        self.stages = []
        stride = stride_init
        for stage in stages:
            name = "layer" + str(stage.index)
            if not stride:
                #当处于stage3~5时, 需要再开始时使用stride=2来downsize
                stride = int(stage.index > 1) + 1
            module = _make_stage(
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
            stride = None
            self.add_module(name, module)
            self.stages.append(name)
        self.out_channels = out_channels

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x

# 创建resnet 的residual-block
def _make_stage(
    transformation_module,
    in_channels,
    bottleneck_channels,
    out_channels,
    block_count,
    num_groups,
    stride_in_1x1,
    first_stride,
    dilation=1,
    dcn_config={}
):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
        )
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)

# 定义每个resnet-bottleneck
# 对于resnet50来说, stage2~5 每个阶段的bottleneck block的数量分别为3,4,6,3
#并且各个相邻stage之间的通道数都是两倍的关系
class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups,
        stride_in_1x1,
        stride,
        dilation,
        norm_func,
        dcn_config
    ):
        super(Bottleneck, self).__init__()
        #downsample: 当bottleneck的输入和输出channels不相等时, 则需要进行下采样.
        self.downsample = None
        if in_channels != out_channels:
            #当输入输出通道数不同时, 额外添加一个1x1的卷积层使得输入通道映射成输出通道数.
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        if dilation > 1:
            stride = 1 # reset to be 1

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)#后接一个固定参数的bn层
        # TODO: specify init for the above
        with_dcn = dcn_config.get("stage_with_dcn", False)
        if with_dcn:#分组卷积
            deformable_groups = dcn_config.get("deformable_groups", 1)
            with_modulated_dcn = dcn_config.get("with_modulated_dcn", False)
            self.conv2 = DFConv2d(
                bottleneck_channels,
                bottleneck_channels,
                with_modulated_dcn=with_modulated_dcn,
                kernel_size=3,
                stride=stride_3x3,
                groups=num_groups,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False
            )
        else:
            self.conv2 = Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=stride_3x3,
                padding=dilation,
                bias=False,
                groups=num_groups,
                dilation=dilation
            )
            nn.init.kaiming_uniform_(self.conv2.weight, a=1)

        self.bn2 = norm_func(bottleneck_channels)

        self.conv3 = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = norm_func(out_channels)

        for l in [self.conv1, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)

        return out

# resnet的第一阶段, 再resnet 50中, 该阶段主要包含一个7x7大小的卷积核,
# 再maskrcnnbenchmark的视线中, 为了方便, 将第二阶段最开始的max pooling层
# 也放在了stem中forward函数中实现( 一般不带参数网络层都放在forward中)
class BaseStem(nn.Module):
    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()

        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        
        #输入channels为3, 输出为64
        self.conv1 = Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_func(out_channels)

        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)#原地激活, 因为不含参数
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

#使用固定的BN
class BottleneckWithFixedBatchNorm(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config={}
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=FrozenBatchNorm2d,
            dcn_config=dcn_config
        )


class StemWithFixedBatchNorm(BaseStem):
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(
            cfg, norm_func=FrozenBatchNorm2d
        )


class BottleneckWithGN(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config={}
    ):
        super(BottleneckWithGN, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=group_norm,
            dcn_config=dcn_config
        )


class StemWithGN(BaseStem):
    def __init__(self, cfg):
        super(StemWithGN, self).__init__(cfg, norm_func=group_norm)

# 文件注册的各个模块, 这些模块会通过配置文件中的字符串信息来决定调用哪一个类或者参数
_TRANSFORMATION_MODULES = Registry({
    "BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm,
    "BottleneckWithGN": BottleneckWithGN,
})

_STEM_MODULES = Registry({
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
    "StemWithGN": StemWithGN,
})

_STAGE_SPECS = Registry({
    "R-50-C4": ResNet50StagesTo4,
    "R-50-C5": ResNet50StagesTo5,
    "R-101-C4": ResNet101StagesTo4,
    "R-101-C5": ResNet101StagesTo5,
    "R-50-FPN": ResNet50FPNStagesTo5,
    "R-50-FPN-RETINANET": ResNet50FPNStagesTo5,
    "R-101-FPN": ResNet101FPNStagesTo5,
    "R-101-FPN-RETINANET": ResNet101FPNStagesTo5,
    "R-152-FPN": ResNet152FPNStagesTo5,
})
```

#### fpn.py

在`backbone.py`文件中的`build_resnet_fpn_backbone(cfg)`函数中, 使用了`fpn=fpn_module.FPN(...)`来创建FPN类的实例化对象, 并且利用`nn.Sequential()`将ResNet和FPN组合在一起形成一个模型, 并将其返回. 实例化代码位于`./maksrcnn_benchmark/modeling/backbone/fpn.py`

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.(实际上为stage2~5的最后一层)
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed-> 指定了送入fpn的每个feature map的通道数
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
                -> 当提供了 top_blocks时, 就会在fpn的最后输出上进行一个额外的op
                    然后result会扩展成result list返回
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        #假设我们使用的是resnet-50-fpn和配置, 则in_channels_list的值为:[256, 512, 1024, 2048]
        for idx, in_channels in enumerate(in_channels_list, 1):
            # 用下标起名: fpn_inner1, fpn_inner_2...
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            # 创建inner_block模块, 这里in_channels为各个stage输出的通道数
            # out_channels为256, 定义在用户配置文件中
            # 这里的卷积核大小为1, 其主要作用是改变通道数到out_channels (降维)
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            # 在当前特征图上添加fpn
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            # 将当前stage的fpn模块的名称添加到对应的列表中
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        # 将top_blocks作为FPN类的成员变量
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
            -> resnet的计算结果正好满足fpn的输入要求, 因此可以使用nn.Sequential直接将两者结合
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
                -> 经过fpn后的特征图组成的列表, 排列顺序是高分辨率的在前
        """
        # 先计算最后一层 (分辨率最低) 特征图的fpn结果
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        # [:-1]获取了前三项, [::-1]代表从头到尾切片, 步长为-1, 效果为列表逆置
        # 举例, zip里的操作self.inner_block[:-1][::-1]的运行结果为
        # [fpn_inner3, fpn_inner2, fpn_inner1], 相当于对列表进行逆置
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            # 根据给定的scale参数对特征图进行放大/缩小, 这里scale=2
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down
            # 将当前satage输出添加到结果列表中, 注意还要用layer_block执行卷积计算
            # 同时为了使得分辨率最大的在前,我们需要将结果插入到0位置.
            results.insert(0, getattr(self, layer_block)(last_inner))

        # 如果top_blocks不为空, 则需要执行如下额外的操作.
        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)# 将新的计算结果追加进列表中.
        # 以元组(只读)形式返回
        return tuple(results)

# 最后一级的max pool层
class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    如果该模型采用retinanet需要采用多的p6和p7层.
    """
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]

```

### roi_heads

当使用backbone和rpn构建特征图谱的生成结构以后, 我们就需要在特征图谱上划分相应的RoI, 该模块的定义入口就是`roi_heads/roi_heads.py`中`build_roi_heads`函数

#### build_roi_heads[入口函数]

```python
def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    # 理论上, 下面的roi可以同时开启, 互不影响, 但通常只会开启其中一个
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
```

- roi_heads/box-head/box_head.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        # 定义在roi_box_feature_extractors.py 文件中
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)#特征提取
        # 函数定义在roi_box_predictors.py
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)# 定义在inference.py
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)# 定义在loss.py

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
```

## 模型定义(modeling)-RPN网络

> 在Faster R-CNN中, 首次提出RPN网络, 该网络用于生成目标检测任务所需要的候选框, 在MaskrcnnBenchmark中, 关于RPN网络的定义位于`./maskrcnn_benchmark/modeling/rpn`文件夹中, 该文件夹包含以下四个文件
>
> `rpn.py, anchor_generator.py, inference.py, loss.py`
>
> 在class GeneralizedRCNN(nn.Module)类中, 会通过`self.rpn = build_rpn(cfg)`函数来创建RPN网络, 该函数位于`./maskrcnn_benchmark/modeling/rpn/rpn.py`

#### rpn.py

```python
def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)

    return RPNModule(cfg, in_channels)
```

构建RPN网络的核心定义在`class RPNModule`中: 

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.rpn.retinanet.retinanet import build_retinanet#todo ??
from .loss import make_rpn_loss_evaluator
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor


class RPNHeadConvRegressor(nn.Module):
    """
    A simple RPN Head for classification and bbox regression
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHeadConvRegressor, self).__init__()
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        assert isinstance(x, (list, tuple))#fpn
        logits = [self.cls_logits(y) for y in x]
        bbox_reg = [self.bbox_pred(y) for y in x]

        return logits, bbox_reg


class RPNHeadFeatureSingleConv(nn.Module):
    """
    Adds a simple RPN Head with one conv to extract the feature
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
        """
        super(RPNHeadFeatureSingleConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        for l in [self.conv]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        self.out_channels = in_channels

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        x = [F.relu(self.conv(z)) for z in x]

        return x

class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and outputs 
    RPN proposals and losses. Works for both FPN and non-FPN.
    从backbone中获取特征图用于计算, 输出proposals和损失值
    """

    def __init__(self, cfg, in_channels):
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()
        #1. 根据配置文件的信息, 输出对应的anchor
        anchor_generator = make_anchor_generator(cfg)
        #2. 创建rpn heads
        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        head = rpn_head(
            cfg, in_channels, anchor_generator.num_anchors_per_location()[0]
        )
        #3. 主要功能是将bounding boxes的表示形式编码成易于训练的形式[cx,cy,dw,dh]
        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        #4. 根据配置信息对候选框进行后处理, 选取合适的框进行训练
        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        # 选取合适的框用于测试
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)
        #5. 利用得到box, 获取损失函数
        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # 利用给定的特征图谱计算相应的rpn结果
        objectness, rpn_box_regression = self.head(features)
        # 在图片上生成anchors
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    #训练状态时, 前向传播函数
    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch. (注意此时不更新网络参数)
            # 对于end-to-end models来说, anchors必须被转为成boxes,
            # 然后采样到目标检测网络的batch中用于训练
            with torch.no_grad():
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets
                )
        #获取损失函数的结果
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            # 对于端到端的模型来说, RPN proposal仅仅时网络的一个中间状态, 无需将它用降序的顺序排序,
            # 之恶极返回RPN结果, 但对于RPN-only的模式, RPN的输出就是最终结果,需要以置信度从高到低
            # 顺序保存结果并返回.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)

    return RPNModule(cfg, in_channels)

```

在`class RPNModule`中, 使用了class RPNHead作为其头部

```python
@registry.RPN_HEADS.register("SingleConvRPNHead")
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        # objectness 预测层, 输出的channels数为anchors的数量. (每一点对应K个anchors)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 预测box回归的网络层
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )
        # 对定义的网络参数进行初始化
        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)
    
    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            #根据卷积+激活后的结果预测objectness
            logits.append(self.cls_logits(t))
            # 根据卷积+激活后的结果预测bbox
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg
```



在定义RPNModule时, 分别使用`make_anchor_generator(), make_rpn_postprocessor()和make_rpn_loss_evaluator()`函数来构建模型的`anchor_generator, box_selector, loss_evaluator`, 这三个函数分别定义在其他的三个文件中, 下面根据函数调用的顺序, 对这几个文件展开解析.

#### anchor_generator.py

生成anchors

```python
# ./maskrcnn_benchmark/modeling/rpn/anchor_generator.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import numpy as np
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        # 获取buffer长度
        return len(self._buffers)

    def __iter__(self):
        # buffer迭代器
        return iter(self._buffers.values())


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    对于给定的一系列image size和feature maps, 计算对应的anchors
    """

    def __init__(
        self,
        sizes=(128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
        anchor_strides=(8, 16, 32),
        straddle_thresh=0,
    ):
        super(AnchorGenerator, self).__init__()

        if len(anchor_strides) == 1:
            anchor_stride = anchor_strides[0]
            cell_anchors = [
                generate_anchors(anchor_stride, sizes, aspect_ratios).float()
            ]
        else:
            if len(anchor_strides) != len(sizes):
                raise RuntimeError("FPN should have #anchor_strides == #sizes")

            cell_anchors = [
                generate_anchors(
                    anchor_stride,
                    size if isinstance(size, (tuple, list)) else (size,),
                    aspect_ratios
                ).float()
                for anchor_stride, size in zip(anchor_strides, sizes)
            ]
        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        # 获取anchors
        anchors = []
        for size, stride, base_anchors in zip(
            grid_sizes, self.strides, self.cell_anchors
        ):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device
            )
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def add_visibility_to(self, boxlist):
        # anchors保留的功能, 如果超出图像是否舍弃
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        if self.straddle_thresh >= 0:
            inds_inside = (
                (anchors[..., 0] >= -self.straddle_thresh)
                & (anchors[..., 1] >= -self.straddle_thresh)
                & (anchors[..., 2] < image_width + self.straddle_thresh)
                & (anchors[..., 3] < image_height + self.straddle_thresh)
            )
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.bool, device=device)
        boxlist.add_field("visibility", inds_inside)

    def forward(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                boxlist = BoxList(
                    anchors_per_feature_map, (image_width, image_height), mode="xyxy"
                )
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors


def make_anchor_generator(config):
    # 根据给定的stride, sizes, aspect_ratio等参数返回一个anchor box组成的矩阵
    anchor_sizes = config.MODEL.RPN.ANCHOR_SIZES
    aspect_ratios = config.MODEL.RPN.ASPECT_RATIOS
    anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH

    if config.MODEL.RPN.USE_FPN:
        assert len(anchor_stride) == len(
            anchor_sizes
        ), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"
    else:
        assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"
    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios, anchor_stride, straddle_thresh
    )
    return anchor_generator


def make_anchor_generator_retinanet(config):
    anchor_sizes = config.MODEL.RETINANET.ANCHOR_SIZES
    aspect_ratios = config.MODEL.RETINANET.ASPECT_RATIOS
    anchor_strides = config.MODEL.RETINANET.ANCHOR_STRIDES
    straddle_thresh = config.MODEL.RETINANET.STRADDLE_THRESH
    octave = config.MODEL.RETINANET.OCTAVE
    scales_per_octave = config.MODEL.RETINANET.SCALES_PER_OCTAVE

    assert len(anchor_strides) == len(anchor_sizes), "Only support FPN now"
    new_anchor_sizes = []
    for size in anchor_sizes:
        per_layer_anchor_sizes = []
        for scale_per_octave in range(scales_per_octave):
            octave_scale = octave ** (scale_per_octave / float(scales_per_octave))
            per_layer_anchor_sizes.append(octave_scale * size)
        new_anchor_sizes.append(tuple(per_layer_anchor_sizes))

    anchor_generator = AnchorGenerator(
        tuple(new_anchor_sizes), aspect_ratios, anchor_strides, straddle_thresh
    )
    return anchor_generator

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#        [-175.,  -87.,  192.,  104.],
#        [-359., -183.,  376.,  200.],
#        [ -55.,  -55.,   72.,   72.],
#        [-119., -119.,  136.,  136.],
#        [-247., -247.,  264.,  264.],
#        [ -35.,  -79.,   52.,   96.],
#        [ -79., -167.,   96.,  184.],
#        [-167., -343.,  184.,  360.]])


def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    # 根据给定的stride, sizes, aspect_ratio等参数返回一个anchor box组成的矩阵
    """
    return _generate_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float),
    )


def _generate_anchors(base_size, scales, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    return torch.from_numpy(anchors)


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    # 返回某个anchor的宽高以及中心坐标
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    给定关于一系列centers的宽和高, 返回对应的anchors
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

```



```python

def make_anchor_generator(config):
    # 根据给定的stride, sizes, aspect_ratio等参数返回一个anchor box组成的矩阵
    # 定义了RPN网络的默认的anchor的面积大小,
    # 默认值为: (32,64,128,512)
    anchor_sizes = config.MODEL.RPN.ANCHOR_SIZES
    # 定义了RPN网络默认的高宽比, 默认值为:(0.5, 1.0, 2.0)
    aspect_ratios = config.MODEL.RPN.ASPECT_RATIOS
    # 定义了RPN网络中feature map采用的stride, 默认值为: (16,)
    anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE
    # 移除那些超过图片STRADDLE_THRESH个像素大小的anchors, 起到剪枝
    # 默认值为0, 如果想要关闭剪枝功能, 则将设置为-1, 或者一个更小的数
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH

    if config.MODEL.RPN.USE_FPN:
        # 当使用fpn时, 要确保rpn和fpn的相关参数匹配
        assert len(anchor_stride) == len(
            anchor_sizes
        ), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"
    else:
        assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"
    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios, anchor_stride, straddle_thresh
    )
    return anchor_generator
```

根据上面函数, `make_anchor_generator(cfg)` 函数会根据对应的配置文件创建一个AnchorGenerator的实例, 因此, 我们就下面对 `class AnchorGenerator(nn.Module)`类进行解析

```python

class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    对于给定的一系列image size和feature maps, 计算对应的anchors
    """

    def __init__(
        self,
        sizes=(128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
        anchor_strides=(8, 16, 32),
        straddle_thresh=0,
    ):
        super(AnchorGenerator, self).__init__()

        if len(anchor_strides) == 1:
            # 如果anchor_strides的长度为1, 说明没有fpn部分, 直接调用相关函数
            anchor_stride = anchor_strides[0]
            cell_anchors = [
                generate_anchors(anchor_stride, sizes, aspect_ratios).float()
            ]
        else:
            if len(anchor_strides) != len(sizes):
                raise RuntimeError("FPN should have #anchor_strides == #sizes")

            cell_anchors = [
                generate_anchors(
                    anchor_stride,
                    size if isinstance(size, (tuple, list)) else (size,),
                    aspect_ratios
                ).float()
                for anchor_stride, size in zip(anchor_strides, sizes)
            ]
        # 及那个strides, cell_anchors, straddle_thresh作为AnchorGenerator的成员
        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        # 获取anchors
        anchors = []
        for size, stride, base_anchors in zip(
            grid_sizes, self.strides, self.cell_anchors
        ):
            grid_height, grid_width = size
            device = base_anchors.device
            # 按照步长来获取偏移量
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device
            )
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device
            )
            #创建关于shifts_y, shifts_x的meshgrid(就是shifts_y x shifts_x的grid)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            #二者展开成一维
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def add_visibility_to(self, boxlist):
        # anchors保留的功能, 如果超出图像是否舍弃
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        if self.straddle_thresh >= 0:
            inds_inside = (
                (anchors[..., 0] >= -self.straddle_thresh)
                & (anchors[..., 1] >= -self.straddle_thresh)
                & (anchors[..., 2] < image_width + self.straddle_thresh)
                & (anchors[..., 3] < image_height + self.straddle_thresh)
            )
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.bool, device=device)
        boxlist.add_field("visibility", inds_inside)

    def forward(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                boxlist = BoxList(
                    anchors_per_feature_map, (image_width, image_height), mode="xyxy"
                )
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors
```

在class AnchorGenerator中, 利用generate_anchors()函数来生成对应的anchors, 该函数是生成anchors的入口, 在生成anchors时, 需要进行一些祭祀啊u你和转换, 其大致流程和对应的实现函数如下所示:

> 1. 获取生成anchors必要的参数, 包括: `stride, sizes, aspect_ratios`. 
>
>    - `stride`: 特征图谱上的anhors的基础尺寸
>    - `sizes`: 代表anchor对应在原始图谱中的大小(以像素为单位), 因此容易知道anchor在特征图谱上的缩放比例为`size/stride`
>    - `aspect_ratios`: 代表anchors的高宽比
>
>    最终返回的anchors的数量就是sizes的数量和aspect_ratios数量的乘积
>
> 2. 获取特征图谱上对应的base_size(stride)后, 我们将其表示成`[x1,y1,x2,y2]`(坐标是相对于anchor的中心而言)的box形式. 
>
> 3. 然后根据`aspect_ratios`的值来获取不同的anchor boxes的尺寸, 例如, stide=4的base_anchors来说, 如果参数aspect_ratios为[0.5, 1.0, 2.0], 那么它就应该返回面积不变, 但是高宽比分别为[0.5, 1.0, 2.0]的三个坐标.
>
> 4. 在获取不同比例的特征图谱上的box坐标以后, 我们就该利用`scales = sizes/stride`来将这些box坐标映射到原始图像中, 也就是按照对应的比例将这些box放大, 对于我们刚刚举得例子`scales=32/4`来说, 最终得box得坐标如下所示. 这部分代码实现位于`_scale_num()`函数中
>
>    `[[-22., -10., 25., 13.], [-14., -14., 17., 17.], [-10., -22., 13., 25.]]`

#### inference.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes

from ..utils import cat
from .utils import permute_and_flatten

class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    主要完成对RPN box的后处理功能 (在将boxes送到heads之前执行)
    """

    def __init__(
        self,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        box_coder=None,
        fpn_post_nms_top_n=None,
        fpn_post_nms_per_batch=True,
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(RPNPostProcessor, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        #创建一个BoxCoder实例
        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.fpn_post_nms_per_batch = fpn_post_nms_per_batch

    def add_gt_proposals(self, proposals, targets):
        """
        将真实的边框标签targets添加到BoxList列表数据中
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].bbox.device

        #将target进行深度复制, gt_boxes是一个列表, 其元素类型为BoxList
        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))
        #调用boxlist_ops.py中的cat_boxlist函数将proposal和gt_box合成一个boxlist
        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        在单一的特征图上执行前向传播, 将anchor转换成box(xyxy)
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = objectness.device
        N, A, H, W = objectness.shape

        # put in the same format as anchors
        # 将格式转换成和anchors相同的格式, 先改变维度的排列, 然后改变shape的形状
        objectness = permute_and_flatten(objectness, N, A, 1, H, W).view(N, -1)
        #sigmoid归一化
        objectness = objectness.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)

        num_anchors = A * H * W

        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        #调用pytorch的topk函数, 该函数返回两个列表, 一个topk的值, 一个是对应下标
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)

        #创建batch的下标, shape为Nx1, 按照顺序递增
        batch_idx = torch.arange(N, device=device)[:, None]
        #获取所有的batch的top_k box
        box_regression = box_regression[batch_idx, topk_idx]

        image_shapes = [box.size for box in anchors]
        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]#一张图片的anchor进行拼接

        proposals = self.box_coder.decode(
            box_regression.view(-1, 4), concat_anchors.view(-1, 4)
        )

        proposals = proposals.view(N, -1, 4)

        result = []
        for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
            #1. 根据当前的结果创建boxlist实例
            boxlist = BoxList(proposal, im_shape, mode="xyxy")
            #2. 添加score
            boxlist.add_field("objectness", score)
            #3. 防止box超出image的边界
            boxlist = boxlist.clip_to_image(remove_empty=False)
            #4. 移除过小的box
            boxlist = remove_small_boxes(boxlist, self.min_size)
            #5. 在当前的box上执行nms算法
            boxlist = boxlist_nms(
                boxlist,
                self.nms_thresh,
                max_proposals=self.post_nms_top_n,
                score_field="objectness",
            )
            result.append(boxlist)
        return result

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
                经过box decoding和NMS操作处理后的anchors
        """
        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)

        # append ground-truth bboxes to proposals
        # 添加gt bboxes到proposal当中去
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)

        return boxlists

    #在所有层次上进行选择
    def select_over_all_levels(self, boxlists):
        #在训练阶段和测试阶段的行为不同, 在训练阶段, post_nms_top_n是在所有的proposals上进行的
        #而在测试阶段, 是在每一个图片的proposal上进行.
        num_images = len(boxlists)
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # NOTE: it should be per image, and not per batch. However, to be consistent 
        # with Detectron, the default is per batch (see Issue #672)
        if self.training and self.fpn_post_nms_per_batch:
            #拼接 'objectness'
            objectness = torch.cat(
                [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
            )
            #获取box数量
            box_sizes = [len(boxlist) for boxlist in boxlists]
            #防止post_nms_top_n超过anchors总数, 产生错误
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            #获取topk的下标
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.bool)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            # 获取所有满足条件的box
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


def make_rpn_postprocessor(config, rpn_box_coder, is_train):
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN#eg: 2000
    if not is_train:# 1000
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST

    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    fpn_post_nms_per_batch = config.MODEL.RPN.FPN_POST_NMS_PER_BATCH
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    #根据配置参数, 创建一个RPNPostProcessor实例
    box_selector = RPNPostProcessor(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        fpn_post_nms_per_batch=fpn_post_nms_per_batch,
    )
    return box_selector

```

#### loss.py

`make_rpn_loss_evaluator()`函数来创建RPN网络的损失函数评价器

```python
def make_rpn_loss_evaluator(cfg, box_coder):
    # 根据配置信息创建matcher实例
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )
    # 根据配置信息创建一个BalancedPositiveNegativeSampler实例
    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )
    # 利用上面创建的实例对象进一步创建RPNLossComputation实例
    loss_evaluator = RPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        generate_rpn_labels
    )
    return loss_evaluator
```

`RPNLossComputation`类的代码实现

## tools

### train_net.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader#数据集载入
from maskrcnn_benchmark.solver import make_lr_scheduler#学习率更新策略
from maskrcnn_benchmark.solver import make_optimizer#设置优化器, 封装pytorch的SGD类
from maskrcnn_benchmark.engine.inference import inference#推演代码
from maskrcnn_benchmark.engine.trainer import do_train#模型训练的核心逻辑
#该函数detectron中的类似, 都是用来创建目标检测模型的, 这也就是模型的入口函数
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
#分布式训练相关设置, 因为我的gpu个数为1, 因此get_rank()会返回0
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
#封装了logging模块, 用于屏幕输出一些日志信息
from maskrcnn_benchmark.utils.logger import setup_logger
#封装了os.mkdirs函数, 当文件夹已存在是会自动略过
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def train(cfg, local_rank, distributed):
    #该语句调用了./maskrcnn_benchmark/modeling/detector/中的build_detection_model()
    #函数, 用来创建目标检测模型, 这也是模型创建的入口函数, 其会根据配置文件返回一个网络模型
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    #封装了torch.optiom.SGD()函数, 根据tensor的requires_grad属性构成需要更新的参数列表
    optimizer = make_optimizer(cfg, model)

    #根据配置信息设置optimizer的学习率更新策略
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    #分布式训练情况下, 并行处理数据
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    #创建一个参数字典, 并将迭代次数设置为0
    arguments = {}
    arguments["iteration"] = 0

    #获取输出的文件夹路径, 默认'.'
    output_dir = cfg.OUTPUT_DIR
    #因为我只有一个gpu, 所以这里save_to_disk=True
    save_to_disk = get_rank() == 0
    #DetectronCheckpointer对象, 后面会有在do_train()函数的参数
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)#加载指定权重文件
    arguments.update(extra_checkpoint_data)#字典的update方法, 对字典的键值进行更新

    #data_loader的类型为列表, 内部元素类型为torch.utils.data.DataLoader
    
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=True)
    else:
        data_loader_val = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
    )

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    #将config_file指定的配置项覆盖到默认配置项当中
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()#冻结所有的配置项, 防止修改

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    #调用train函数.
    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()

```

#### trainer.py

- do_train()

  ```python
  # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
  import datetime
  import logging
  import os
  import time
  
  import torch
  import torch.distributed as dist
  from tqdm import tqdm
  
  from maskrcnn_benchmark.data import make_data_loader
  from maskrcnn_benchmark.utils.comm import get_world_size, synchronize
  from maskrcnn_benchmark.utils.metric_logger import MetricLogger
  from maskrcnn_benchmark.engine.inference import inference
  
  # from apex import amp
  
  def reduce_loss_dict(loss_dict):
      """
      Reduce the loss dictionary from all processes so that process with rank
      0 has the averaged results. Returns a dict with the same fields as
      loss_dict, after reduction.
      """
      world_size = get_world_size()
      if world_size < 2:
          return loss_dict
      with torch.no_grad():
          loss_names = []
          all_losses = []
          for k in sorted(loss_dict.keys()):
              loss_names.append(k)
              all_losses.append(loss_dict[k])
          all_losses = torch.stack(all_losses, dim=0)
          dist.reduce(all_losses, dst=0)
          if dist.get_rank() == 0:
              # only main process gets accumulated, so only divide by
              # world_size in this case
              all_losses /= world_size
          reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
      return reduced_losses
  
  
  def do_train(
      cfg,
      model,
      data_loader,
      data_loader_val,
      optimizer,
      scheduler,#学习更新策略, 封装在solver/lr_scheduler.py
      checkpointer,#DetectronCheckpointer, 用于自动转化caffe2 Detectron的模型文件
      device,
      checkpoint_period,#指定模型的保存迭代间隔
      test_period,
      arguments,#额外的其他参数, 字典类型, 一般情况只有arguments[iteratioin], 初值为0
  ):
      logger = logging.getLogger("maskrcnn_benchmark.trainer")#记录日志信息
      logger.info("Start training")
      #用于记录一些变量的滑动平均值和全局平均值
      meters = MetricLogger(delimiter="  ")#delimiter为定界符,
  
      #数据载入器重写了len函数, 使其返回载入器需要提供的batch的次数, 即cfg.SOLVER.MAX_ITER
      max_iter = len(data_loader)
      start_iter = arguments["iteration"]#默认为0, 但是会根据载入的权重文件, 变成其他值.
      model.train()
      start_training_time = time.time()
      end = time.time()
  
      iou_types = ("bbox",)
      if cfg.MODEL.MASK_ON:
          iou_types = iou_types + ("segm",)
      if cfg.MODEL.KEYPOINT_ON:
          iou_types = iou_types + ("keypoints",)
      dataset_names = cfg.DATASETS.TEST
  
      loss_list = []
      lr_list = []
  
      #遍历data_loader, 第二个参数是设置序号的开始序号.
      for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
          if any(len(target) < 1 for target in targets):
              logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
              continue
          data_time = time.time() - end#获取一个batch所需的时间
          iteration = iteration + 1
          arguments["iteration"] = iteration
  
          images = images.to(device)
          targets = [target.to(device) for target in targets]
  
          loss_dict = model(images, targets)#根据image和targets计算loss
  
          losses = sum(loss for loss in loss_dict.values())#将各个loss合并
  
          # reduce losses over all GPUs for logging purposes
          loss_dict_reduced = reduce_loss_dict(loss_dict)
          losses_reduced = sum(loss for loss in loss_dict_reduced.values())
          meters.update(loss=losses_reduced, **loss_dict_reduced)#更新滑动平均值
  
          optimizer.zero_grad()#清除梯度缓存
          # Note: If mixed precision is not used, this ends up doing nothing
          # Otherwise apply loss scaling for mixed-precision recipe
          # with amp.scale_loss(losses, optimizer) as scaled_losses:
          #     scaled_losses.backward()#todo 被注释掉
          losses.backward()#计算梯度
          optimizer.step()#更新参数
          scheduler.step()#更新一次学习率
  
          batch_time = time.time() - end
          end = time.time()
          meters.update(time=batch_time, data=data_time)
  
          #根据时间的滑动平均值计算大约还剩多长时间结束训练
          eta_seconds = meters.time.global_avg * (max_iter - iteration)
          eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
  
          #每经过20此迭代, 输出一次训练状态
          if iteration % 20 == 0 or iteration == max_iter:
              loss_list.append(losses_reduced.item())
              lr_list.append(optimizer.param_groups[0]["lr"])
              logger.info(
                  meters.delimiter.join(
                      [
                          "eta: {eta}",
                          "iter: {iter}",
                          "{meters}",
                          "lr: {lr:.6f}",
                          "max mem: {memory:.0f}",
                      ]
                  ).format(
                      eta=eta_string,
                      iter=iteration,
                      meters=str(meters),
                      lr=optimizer.param_groups[0]["lr"],
                      memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                  )
              )
          #模型保存
          if iteration % checkpoint_period == 0:
              checkpointer.save("model_{:07d}".format(iteration), **arguments)
  
          #验证集验证
          if data_loader_val is not None and test_period > 0 and iteration % test_period == 0:
              meters_val = MetricLogger(delimiter="  ")
              synchronize()
              _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                  model,
                  # The method changes the segmentation mask format in a data loader,
                  # so every time a new data loader is created:
                  make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1), is_for_period=True),
                  dataset_name="[Validation]",
                  iou_types=iou_types,
                  box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                  device=cfg.MODEL.DEVICE,
                  expected_results=cfg.TEST.EXPECTED_RESULTS,
                  expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                  output_folder=None,
              )
              synchronize()
              model.train()
              with torch.no_grad():
                  # Should be one image for each GPU:
                  for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                      images_val = images_val.to(device)
                      targets_val = [target.to(device) for target in targets_val]
                      loss_dict = model(images_val, targets_val)
                      losses = sum(loss for loss in loss_dict.values())
                      loss_dict_reduced = reduce_loss_dict(loss_dict)
                      losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                      meters_val.update(loss=losses_reduced, **loss_dict_reduced)
              synchronize()
              logger.info(
                  meters_val.delimiter.join(
                      [
                          "[Validation]: ",
                          "eta: {eta}",
                          "iter: {iter}",
                          "{meters}",
                          "lr: {lr:.6f}",
                          "max mem: {memory:.0f}",
                      ]
                  ).format(
                      eta=eta_string,
                      iter=iteration,
                      meters=str(meters_val),
                      lr=optimizer.param_groups[0]["lr"],
                      memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                  )
              )
          #达到最大迭代次数后, 也进行保存.
          if iteration == max_iter:
              checkpointer.save("model_final", **arguments)
      
      #输出总的训练耗时
      total_training_time = time.time() - start_training_time
      total_time_str = str(datetime.timedelta(seconds=total_training_time))
      logger.info(
          "Total training time: {} ({:.4f} s / it)".format(
              total_time_str, total_training_time / (max_iter)
          )
      )
      print(loss_list,lr_list)
      from plot_curve import plot_loss_and_lr
      with open('loss_result.txt','w') as f:
          for i in range(len(loss_list)):
              f.write("{} {} {} \n".format(i,loss_list[i],lr_list[i]))
  
      plot_loss_and_lr(loss_list,lr_list)
  
  ```

### test_net.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    #权重文件路径
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    #其他配置选项
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    #将指定的配置文件的设置覆盖到全局设置中
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()#冻结配置信息, 防止更改

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    #根据配置信息创建模型
    model = build_detection_model(cfg)
    #将模型移动到指定设备上
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    #获取输出文件夹的父路径
    output_dir = cfg.OUTPUT_DIR
    #加载权重
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    #设置iou类型
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    #根据数据集的数量定义输出文件夹
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    
    #创建输出文件夹
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    
    #加载测试数据集
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    
    #对数据集中的数据按批次调用inference函数
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


if __name__ == "__main__":
    main()

```

#### inference.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug

#计算结果, 获得预测结果
def compute_on_dataset(model, data_loader, device, bbox_aug, timer=None):
    model.eval()#将模型的状态设置为eval, 主要影响dropout, bn等操作的行为
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        # images = images.tensors#转为list
        with torch.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        # 更新结果字典
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict

# 累积预测
def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,#从build_detection_model函数得到模型对象
        data_loader,
        dataset_name,#str, 数据集名称
        iou_types=("bbox",),
        box_only=False,
        bbox_aug=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,#自定义输出文件夹
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()#获取设备树
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    #调用本文件的函数, 获取预测结果
    predictions = compute_on_dataset(model, data_loader, device, bbox_aug, inference_timer)
    # wait for all processes to complete before measuring the time
    #等到所有的进程都结束以后, 再计算总耗时
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )
    
    #将所有GPU设备上的预测结果累加并返回
    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
    #调用评价函数, 返回预测结果的质量
    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)

```

## structures

定义了检测模式下包含的数据结构

### bounding_box.py

定义`class BoxList(object)`类, 该类用于表示一系列的bounding boxes. 这些boxes会以N*4大小的tensor来表示, 为了唯一确定的boxes在图片中的准确位置, 该类还保存了图片的维度, 另外也可以添加额外的信息到特定bouding box中, 如标签信息

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="xyxy"):
        #根据bbox的数据类型, 获取对应的device
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        #将box数据类型转换成tensor类型
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        #bbox的维度数量必须为2, 并且第二位必须为4, 即shape=(n,4)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 4:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    #添加新的键值或覆盖旧的键值
    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    #获取指定键对应的值
    def get_field(self, field):
        return self.extra_fields[field]
    #判断额外信息中是否存在该键
    def has_field(self, field):
        return field in self.extra_fields
    #以列表的形式返回所有键的名称
    def fields(self):
        return list(self.extra_fields.keys())
    #将另一个boxlist类型的额外信息(字典)复制到额外信息(extra_fields)中
    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v
    #将当前box的表示形式转换成参数指定的模式
    def convert(self, mode):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 1
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    #获取bbox的(x1,y1, x2,y2)形式的坐标表示
    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    #将所有的boxes按照给定的size和图片尺寸进行缩放, 创建一个副本存储缩放后的boxes并返回
    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            #令所有的box都乘以缩放比, 不论box是xyxy形式还是xywh表示, \
            #乘以系数就可以正确的将box的坐标转换到缩放后图片的对应坐标
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

        #宽和高的缩放比不通过, 因此需要拆分后分别放缩然后连接在一起
        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )
        bbox = BoxList(scaled_box, size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        对bbox进行转化(翻转或者选择90度)
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        transposed_boxes = torch.cat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )
        #根据转换后的boxes坐标创建一个新的BoxList实例, 同时将extra_fields信息复制
        bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        #将box的mode转换后返回
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Crops a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        box是一个4元组, 指定了希望裁剪的区域的左上角和右下角
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox = BoxList(cropped_box, (w, h), mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    # Tensor-like methods

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        #深度复制函数
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


if __name__ == "__main__":
    bbox = BoxList([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)

```

### bounding_opt.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .bounding_box import BoxList

from maskrcnn_benchmark.layers import nms as _box_nms

#会对一个boxlist类型数据中的box执行非极大值抑制算法
def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode#缓存当前的模式
    boxlist = boxlist.convert("xyxy")#转换成指定模式
    boxes = boxlist.bbox#获取n*4的bbox列表
    score = boxlist.get_field(score_field)#获取对应的score列表
    #调用box_nms执行非极大值抑制
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size
    使得boxlist只保留那些尺寸大于一定值的box
    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero(as_tuple=False).squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

#该函数会将一个组成元素为boxlist的列表合并成一个boxlist对象
def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    #确保类型为列表或者元组, 且其中的元素类型为boxlist
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)
    
    #确保所有的boxlist的size, model以及extra_fields字典的keys是相同的
    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)
    
    #调用本文件的_cat()方法, 将bboxes里面的boxlist数据连接成一个boxlist
    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes

```

## RoI

### box_head_loss

### keypoints_head_loss

### mask_head_loss
