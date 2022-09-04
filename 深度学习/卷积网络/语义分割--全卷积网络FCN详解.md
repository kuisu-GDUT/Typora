# 语义分割--全卷积网络FCN详解

在分类使用的网络通常会在最后连接几层全连接, 这会将原来二维的矩阵(图片)压缩成一维, 从而丢失空间信息. 而图像语义分割的输出需要是分割图/

![image-20210930100318316](E:\kuisu\typora\深度学习资料\卷积网络\语义分割--全卷积网络FCN详解.assets\image-20210930100318316-16329673999461.png)

FCN原理: FCN将传统卷积网络后面的全连接层换成了卷积层, 这样网络输出的是heatmap; 同时为解决因卷积核池化对图像尺寸的影响, 提出使用上采样的方式恢复.

![image-20210930100629988](E:\kuisu\typora\深度学习资料\卷积网络\语义分割--全卷积网络FCN详解.assets\image-20210930100629988-16329675912932.png)

反卷积(deconvolutional)

反卷积核卷积类似, 都是相乘相加的运算, 只不过后者是多对一, 前者是一对多. 而反卷积的前向和后向传播, 只用颠倒卷积的前后向传播即可, 所以无论是优化还是反向传播算法都没有问题.

![img](E:\kuisu\typora\深度学习资料\卷积网络\语义分割--全卷积网络FCN详解.assets\70)

![image-20210930101658655](E:\kuisu\typora\深度学习资料\卷积网络\语义分割--全卷积网络FCN详解.assets\image-20210930101658655-16329682201274.png)

损失计算

![image-20210930101750661](E:\kuisu\typora\深度学习资料\卷积网络\语义分割--全卷积网络FCN详解.assets\image-20210930101750661-16329682726075.png)

- 结果

![image-20211008134642928](E:\kuisu\typora\深度学习资料\卷积网络\语义分割--全卷积网络FCN详解.assets\image-20211008134642928-16336720054622.png)

## FCN网络构建

![《全卷积网络（FCN）详解》](E:\kuisu\typora\深度学习资料\卷积网络\语义分割--全卷积网络FCN详解.assets\MdTPSS.png)

```python

class FCNs(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)#stride=2, 表示上采样2
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1) 
        # classifier is 1x1 conv, to reduce channels from 32 to n_class

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  
        x4 = output['x4']  
        x3 = output['x3']  
        x2 = output['x2']  
        x1 = output['x1']  

        score = self.bn1(self.relu(self.deconv1(x5))) #将x5进行上采样.    
        score = score + x4#将上采样后的x5和x4进行相加
        score = self.bn2(self.relu(self.deconv2(score)))  
        score = score + x3
        score = self.bn3(self.relu(self.deconv3(score)))  
        score = score + x2
        score = self.bn4(self.relu(self.deconv4(score)))  
        score = score + x1
        score = self.bn5(self.relu(self.deconv5(score)))  
        score = self.classifier(score)                    

        return score  
```

### 重写VGG网络

```python
#此处继承所有的VGG
class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(features=make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        # delete redundant fully-connected layer params, can save memory
        # 去掉vgg最后的全连接层(classifier)
        if remove_fc:  
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx, (begin, end) in enumerate(self.ranges):
        #self.ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)) (vgg16 examples)
            for layer in range(begin, end):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output
```

#配置不同大小的VGG

```python
ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# Vgg-Net config 
# Vgg网络结构配置
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# 由cfg构建vgg-Net
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
```

