# Torchvision object detection finetuning tutorial

for this tutorial, we will be finetuning a pre-trained **Mask R-CNN** model in the **Penn-Fudan Database for Pedestrian Detection and Segmentation**. It contains 170 images with 345 instances of pedestrians, and we will used it to illustrate how to use the new features in torchvision in order to train an instance segmentation model on a custom dataset

## Defining the Dataset

the dataset should inherit from the standard `torch.utils.data.Dataset` class, and implement `__len__` and `__getitem__`.

The only specificity that we require is that the dataset `__getitem__` should return:

- imge: a PIL Image of size `(H,W)`
- target: a dict containing the following fieds
  - `boxes (FloatTensor[N, 4])`: the coordinates of the `N` bounding boxes in `[x0,y0,x1,y1]` format, ranging from `0 to W` and `o to W`.
  - `labels(Int64Tensor[N])`: the label for each bouding box. `0` represents always the background class.
  - `image_id(Int64Tensor[1])`: an image identifier. It should be unique between all the images in the dataset,  and is used during evaluation
  - `area(Tensor[N])`:





## 参考

https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html