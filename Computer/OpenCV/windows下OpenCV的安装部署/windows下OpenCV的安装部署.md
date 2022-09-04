# Windows下OpenCV的安装部署教程

## 简介

[OpenCV](https://opencv.org/releases/), 全称Open Source Computer Vision Library, 一个跨平台的计算机视觉库, 以BSD许可证授权发行, 可以在商业和研究领域免费试用.

OpenCV用C++语言编写, 主要接口时C++语言. 

简单理解OpenCV就是一个库, 一个SDK, 一个开发包, 解压后直接可以使用.

## 下载

到[OpenCV](https://opencv.org/releases/)官网下载需要的版本, 在点击Library.

![在这里插入图片描述](.\windows下OpenCV的安装部署.assets\2021070314062318.png#pic_center)

OpenCV支持很多平台.

![image-20211206154519839](.\windows下OpenCV的安装部署.assets\image-20211206154519839-16387767209042.png)

点击windows后, 弹出下载界面, 等待5s后开始自动下载.

![这里写图片描述](.\windows下OpenCV的安装部署.assets\20180807093013393)

然后双击解压, 得到一个文件夹, 选择适当的安装位置.

![这里写图片描述](.\windows下OpenCV的安装部署.assets\20180807101015789)

![这里写图片描述](.\windows下OpenCV的安装部署.assets\20180807101038807)

解压完打开文件夹如下:

![这里写图片描述](.\windows下OpenCV的安装部署.assets\20180807101236346)

其中`build` 是OpenCV使用时要用的一些库文件, 而source中则是OpenCV官方为我们提供的一些demo示例源码

## 配置环境变量

把OpenCV文件夹放好后, 依次选择计算机->属性->高级系统->环境变量, 找到Path变量, 选择并点击编辑, 然后将OpenCV执行文件的路径填入.

![这里写图片描述](.\windows下OpenCV的安装部署.assets\20180807101427956)

OpenCV执行文件路径如下

![image-20211206155018467](.\windows下OpenCV的安装部署.assets\image-20211206155018467-16387770195608.png)

## 部署OpenCV

### 打开visual studio

新建控制台工程

### 添加包含目录

首先要在“解决方案”中选中你的项目名称，如图中绿色框所示。
然后，依次选择 项目—>属性—>VC++目录—>包含目录—>编辑
找到你的包含目录添加就可以了，最好添加三个，我的是这样的：

`D:\Program\opencv\opencv\build\include`

`D:\Program\opencv\opencv\build\include\opencv2`

![在这里插入图片描述](.\windows下OpenCV的安装部署.assets\20210503163947435.png#pic_center)

### 添加库目录

依次选择项目->属性->VC++目录->库目录->编辑

`D:\Program\opencv\opencv\build\x64\vc15\lib`

![这里写图片描述](.\windows下OpenCV的安装部署.assets\20180807111831210)

### 添加附加依赖项

依次选择项目->属性->链接器->输入->附加依赖项->编辑

添加对应的库文件名:`opencv_world454d.lib`

![这里写图片描述](.\windows下OpenCV的安装部署.assets\20180807112228800)

库文件的位置

![这里写图片描述](.\windows下OpenCV的安装部署.assets\20180807112415822)

有两个文件opencv_world341d.lib和opencv_world341.lib
　　如果配置为Debug，选择opencv_world341d.lib
　　如果为Release，选择opencv_world341.lib
　　这里注意，如果你下载的是OpenCV2.x版本，这里的库文件比较多，都填进去就可以了。

## 运行示例

```c++
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(void)
{
	Mat originalImage = imread("E:\\kuisu\\vsStudio\\OpenCV\\cat.jpg");
	if (originalImage.empty())
	{
		cout << "fail to load image !" << endl;
		return -1;
	}
	namedWindow("opencv test", WINDOW_AUTOSIZE);
	imshow("opencv test", originalImage);
	waitKey(0);
	return 0;
}
```

![image-20211206161218484](.\windows下OpenCV的安装部署.assets\image-20211206161218484-163877833934713.png)