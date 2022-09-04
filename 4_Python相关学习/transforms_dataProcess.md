# transform+dataProcess

## transforms

### toTensor()

![在这里插入图片描述](E:\kuisu\typora\Python学习记录\transforms_dataProcess.assets\toTensor.png)

如果是PIL的图像的像素进行归一化[0-1]之间, 并将图像维度进行变化为[c,w,h], 最后转为tensor格式.



### Normalize()

- 对于mean和std参数的一些解惑

  1. 很多代码都是这样写: `torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])`

     这组平均数和std是从imagenet训练集中抽样算出来.

  2. mean, std需要在normalize之前自己先算好再传进去, 不然每次normalize程序就得把所有得图片都读取一篇算这两个数.

  对于imagenet数据集, 其在加载得时候就已转化成[0,1]



![在这里插入图片描述](E:\kuisu\typora\Python学习记录\transforms_dataProcess.assets\20191228122243195.png)

Normalize()是对数据按通道进行标准化

数据如果分布在(0,1)之间, 可能试剂得bias会比较大, 二模型初始化时b=0的, 这样会导致网络收敛比较慢, 经过normalize后, 可以加快模型的收敛速度

### ColorJitter

可以改变图像的属性, 如亮度(brightness), 对比度(contrast),饱和度(saturation),色调(hue)

```python
brightness_change = transforms.ColorJitter(brightness=0.5)
#将图像的亮度随机变化为原图亮度的50%~150%
```

- 示例

  ```python
  import torchvision.transforms as transforms
  
  #单独设置
  #1. 随机改变图像的亮度
  brightness_change = transforms.ColorJitter(brightness=0.5)
  #2. 随机改变图像的色调
  hue_change = transforms.ColorJitter(hue=0.5)
  #3. 随机改变图像的对比度
  contrast_change = transforms.ColorJitter(contrast=0.5)
  
  #综合设置
  color_aug = transforms.ColorJitter(brightness=0.5,
                                    contrast=0.5,
                                    hue=0.5)
  
  transform = transforms.Compose([
      brightness_change,
      hue_change,
      contrast_change
  ])
  ```

### 图像变换

#### Image模块+target模块

```python
import random

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target

class ToPILImage(object):
    def __init__(self,mean=0.5,std=0.5):
        self.mean = mean
        self.std = std
    def __call__(self, image,target=None):
        PILImage = torchvision.transforms.ToPILImage()
        image = image*self.std+self.mean
        if target is None:
            return PILImage(image)
        return PILImage(image),target
```

### BoxList模块

```python
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
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
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

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

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

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

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
        bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Crops a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
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
```

### keypoints模块

```python
class Keypoints(object):
    def __init__(self, keypoints, size, mode=None):
        # FIXME remove check once we have better integration with device
        # in my version this would consistently return a CPU tensor
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device('cpu')
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
        num_keypoints = keypoints.shape[0]
        if num_keypoints:
            keypoints = keypoints.view(num_keypoints, -1, 3)
        
        # TODO should I split them?
        # self.visibility = keypoints[..., 2]
        self.keypoints = keypoints# [..., :2]

        self.size = size
        self.mode = mode
        self.extra_fields = {}

    def crop(self, box):
        raise NotImplementedError()

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        ratio_w, ratio_h = ratios
        resized_data = self.keypoints.clone()
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
        keypoints = type(self)(resized_data, size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError(
                    "Only FLIP_LEFT_RIGHT implemented")

        flip_inds = type(self).FLIP_INDS
        flipped_data = self.keypoints[:, flip_inds]
        width = self.size[0]
        TO_REMOVE = 1
        # Flip x coordinates
        flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE

        # Maintain COCO convention that if visibility == 0, then x, y = 0
        inds = flipped_data[..., 2] == 0
        flipped_data[inds] = 0

        keypoints = type(self)(flipped_data, self.size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def to(self, *args, **kwargs):
        keypoints = type(self)(self.keypoints.to(*args, **kwargs), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            keypoints.add_field(k, v)
        return keypoints

    def __getitem__(self, item):
        keypoints = type(self)(self.keypoints[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v[item])
        return keypoints

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.keypoints))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s

```

### imageList模块

```python
class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)


def to_image_list(tensors, size_divisible=0):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]

    if isinstance(tensors, ImageList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        # single tensor shape can be inferred
        if tensors.dim() == 3:
            tensors = tensors[None]
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return ImageList(tensors, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        # TODO Ideally, just remove this and let me model handle arbitrary
        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new(*batch_shape).zero_()
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        image_sizes = [im.shape[-2:] for im in tensors]

        return ImageList(batched_imgs, image_sizes)
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))

```

### mask模块

#### BinaryMaskList

```python
class BinaryMaskList(object):
    """
    This class handles binary masks for all objects in the image
    """

    def __init__(self, masks, size):
        """
            Arguments:
                masks: Either torch.tensor of [num_instances, H, W]
                    or list of torch.tensors of [H, W] with num_instances elems,
                    or RLE (Run Length Encoding) - interpreted as list of dicts,
                    or BinaryMaskList.
                size: absolute image size, width first

            After initialization, a hard copy will be made, to leave the
            initializing source data intact.
        """

        assert isinstance(size, (list, tuple))
        assert len(size) == 2

        if isinstance(masks, torch.Tensor):
            # The raw data representation is passed as argument
            masks = masks.clone()
        elif isinstance(masks, (list, tuple)):
            if len(masks) == 0:
                masks = torch.empty([0, size[1], size[0]])  # num_instances = 0!
            elif isinstance(masks[0], torch.Tensor):
                masks = torch.stack(masks, dim=0).clone()
            elif isinstance(masks[0], dict) and "counts" in masks[0]:
                if(isinstance(masks[0]["counts"], (list, tuple))):
                    masks = mask_utils.frPyObjects(masks, size[1], size[0])
                # RLE interpretation
                rle_sizes = [tuple(inst["size"]) for inst in masks]

                masks = mask_utils.decode(masks)  # [h, w, n]
                masks = torch.tensor(masks).permute(2, 0, 1)  # [n, h, w]#有多少各二值mask

                assert rle_sizes.count(rle_sizes[0]) == len(rle_sizes), (
                    "All the sizes must be the same size: %s" % rle_sizes
                )

                # in RLE, height come first in "size"
                rle_height, rle_width = rle_sizes[0]
                assert masks.shape[1] == rle_height
                assert masks.shape[2] == rle_width

                width, height = size
                if width != rle_width or height != rle_height:
                    masks = interpolate(
                        input=masks[None].float(),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )[0].type_as(masks)
            else:
                RuntimeError(
                    "Type of `masks[0]` could not be interpreted: %s"
                    % type(masks)
                )
        elif isinstance(masks, BinaryMaskList):
            # just hard copy the BinaryMaskList instance's underlying data
            masks = masks.masks.clone()
        else:
            RuntimeError(
                "Type of `masks` argument could not be interpreted:%s"
                % type(masks)
            )

        if len(masks.shape) == 2:
            # if only a single instance mask is passed
            masks = masks[None]

        assert len(masks.shape) == 3
        assert masks.shape[1] == size[1], "%s != %s" % (masks.shape[1], size[1])
        assert masks.shape[2] == size[0], "%s != %s" % (masks.shape[2], size[0])

        self.masks = masks
        self.size = tuple(size)

    def transpose(self, method):
        dim = 1 if method == FLIP_TOP_BOTTOM else 2
        flipped_masks = self.masks.flip(dim)
        return BinaryMaskList(flipped_masks, self.size)

    def crop(self, box):
        assert isinstance(box, (list, tuple, torch.Tensor)), str(type(box))
        # box is assumed to be xyxy
        current_width, current_height = self.size
        xmin, ymin, xmax, ymax = [round(float(b)) for b in box]

        assert xmin <= xmax and ymin <= ymax, str(box)
        xmin = min(max(xmin, 0), current_width - 1)
        ymin = min(max(ymin, 0), current_height - 1)

        xmax = min(max(xmax, 0), current_width)
        ymax = min(max(ymax, 0), current_height)

        xmax = max(xmax, xmin + 1)
        ymax = max(ymax, ymin + 1)

        width, height = xmax - xmin, ymax - ymin
        cropped_masks = self.masks[:, ymin:ymax, xmin:xmax]
        cropped_size = width, height
        return BinaryMaskList(cropped_masks, cropped_size)

    def resize(self, size):
        try:
            iter(size)
        except TypeError:
            assert isinstance(size, (int, float))
            size = size, size
        width, height = map(int, size)

        assert width > 0
        assert height > 0

        # Height comes first here!
        resized_masks = interpolate(
            input=self.masks[None].float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0].type_as(self.masks)
        resized_size = width, height
        return BinaryMaskList(resized_masks, resized_size)

    def convert_to_polygon(self):
        if self.masks.numel() == 0:
            return PolygonList([], self.size)

        contours = self._findContours()
        return PolygonList(contours, self.size)

    def to(self, *args, **kwargs):
        return self

    def _findContours(self):
        contours = []
        masks = self.masks.detach().numpy()
        for mask in masks:
            mask = cv2.UMat(mask)#将mask转为CV2的格式
            contour, hierarchy = cv2_util.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
            )

            reshaped_contour = []
            for entity in contour:
                assert len(entity.shape) == 3
                assert (
                    entity.shape[1] == 1
                ), "Hierarchical contours are not allowed"
                reshaped_contour.append(entity.reshape(-1).tolist())
            contours.append(reshaped_contour)
        return contours

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        if self.masks.numel() == 0:
            raise RuntimeError("Indexing empty BinaryMaskList")
        return BinaryMaskList(self.masks[index], self.size)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.masks))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s
```

#### PolygonInstance

```python
class PolygonInstance(object):
    """
    This class holds a set of polygons that represents a single instance
    of an object mask. The object can be represented as a set of
    polygons
    """

    def __init__(self, polygons, size):
        """
            Arguments:
                a list of lists of numbers.
                The first level refers to all the polygons that compose the
                object, and the second level to the polygon coordinates.
        """
        if isinstance(polygons, (list, tuple)):
            valid_polygons = []
            for p in polygons:
                p = torch.as_tensor(p, dtype=torch.float32)
                if len(p) >= 6:  # 3 * 2 coordinates
                    valid_polygons.append(p)
            polygons = valid_polygons

        elif isinstance(polygons, PolygonInstance):
            polygons = copy.copy(polygons.polygons)

        else:
            RuntimeError(
                "Type of argument `polygons` is not allowed:%s"
                % (type(polygons))
            )

        """ This crashes the training way too many times...
        for p in polygons:
            assert p[::2].min() >= 0
            assert p[::2].max() < size[0]
            assert p[1::2].min() >= 0
            assert p[1::2].max() , size[1]
        """

        self.polygons = polygons
        self.size = tuple(size)

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        flipped_polygons = []
        width, height = self.size
        if method == FLIP_LEFT_RIGHT:
            dim = width
            idx = 0
        elif method == FLIP_TOP_BOTTOM:
            dim = height
            idx = 1

        for poly in self.polygons:
            p = poly.clone()
            TO_REMOVE = 1
            p[idx::2] = dim - poly[idx::2] - TO_REMOVE
            flipped_polygons.append(p)

        return PolygonInstance(flipped_polygons, size=self.size)

    def crop(self, box):
        assert isinstance(box, (list, tuple, torch.Tensor)), str(type(box))

        # box is assumed to be xyxy
        current_width, current_height = self.size
        xmin, ymin, xmax, ymax = map(float, box)

        assert xmin <= xmax and ymin <= ymax, str(box)
        xmin = min(max(xmin, 0), current_width - 1)
        ymin = min(max(ymin, 0), current_height - 1)

        xmax = min(max(xmax, 0), current_width)
        ymax = min(max(ymax, 0), current_height)

        xmax = max(xmax, xmin + 1)
        ymax = max(ymax, ymin + 1)

        w, h = xmax - xmin, ymax - ymin

        cropped_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            p[0::2] = p[0::2] - xmin  # .clamp(min=0, max=w)
            p[1::2] = p[1::2] - ymin  # .clamp(min=0, max=h)
            cropped_polygons.append(p)

        return PolygonInstance(cropped_polygons, size=(w, h))

    def resize(self, size):
        try:
            iter(size)
        except TypeError:
            assert isinstance(size, (int, float))
            size = size, size

        ratios = tuple(
            float(s) / float(s_orig) for s, s_orig in zip(size, self.size)
        )

        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_polys = [p * ratio for p in self.polygons]
            return PolygonInstance(scaled_polys, size)

        ratio_w, ratio_h = ratios
        scaled_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h
            scaled_polygons.append(p)

        return PolygonInstance(scaled_polygons, size=size)

    def convert_to_binarymask(self):
        width, height = self.size
        # formatting for COCO PythonAPI
        polygons = [p.numpy() for p in self.polygons]
        rles = mask_utils.frPyObjects(polygons, height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle)
        mask = torch.from_numpy(mask)
        return mask

    def __len__(self):
        return len(self.polygons)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_groups={}, ".format(len(self.polygons))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s
```

#### PolygonList

```python
class PolygonList(object):
    """
    This class handles PolygonInstances for all objects in the image
    """

    def __init__(self, polygons, size):
        """
        Arguments:
            polygons:
                a list of list of lists of numbers. The first
                level of the list correspond to individual instances,
                the second level to all the polygons that compose the
                object, and the third level to the polygon coordinates.

                OR

                a list of PolygonInstances.

                OR

                a PolygonList

            size: absolute image size

        """
        if isinstance(polygons, (list, tuple)):
            if len(polygons) == 0:
                polygons = [[[]]]
            if isinstance(polygons[0], (list, tuple)):
                assert isinstance(polygons[0][0], (list, tuple)), str(
                    type(polygons[0][0])
                )
            else:
                assert isinstance(polygons[0], PolygonInstance), str(
                    type(polygons[0])
                )

        elif isinstance(polygons, PolygonList):
            size = polygons.size
            polygons = polygons.polygons

        else:
            RuntimeError(
                "Type of argument `polygons` is not allowed:%s"
                % (type(polygons))
            )

        assert isinstance(size, (list, tuple)), str(type(size))

        self.polygons = []
        for p in polygons:
            p = PolygonInstance(p, size)
            if len(p) > 0:
                self.polygons.append(p)

        self.size = tuple(size)

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        flipped_polygons = []
        for polygon in self.polygons:
            flipped_polygons.append(polygon.transpose(method))

        return PolygonList(flipped_polygons, size=self.size)

    def crop(self, box):
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_polygons = []
        for polygon in self.polygons:
            cropped_polygons.append(polygon.crop(box))

        cropped_size = w, h
        return PolygonList(cropped_polygons, cropped_size)

    def resize(self, size):
        resized_polygons = []
        for polygon in self.polygons:
            resized_polygons.append(polygon.resize(size))

        resized_size = size
        return PolygonList(resized_polygons, resized_size)

    def to(self, *args, **kwargs):
        return self

    def convert_to_binarymask(self):
        if len(self) > 0:
            masks = torch.stack(
                [p.convert_to_binarymask() for p in self.polygons]
            )
        else:
            size = self.size
            masks = torch.empty([0, size[1], size[0]], dtype=torch.bool)

        return BinaryMaskList(masks, size=self.size)

    def __len__(self):
        return len(self.polygons)

    def __getitem__(self, item):
        if isinstance(item, int):
            selected_polygons = [self.polygons[item]]
        elif isinstance(item, slice):
            selected_polygons = self.polygons[item]
        else:
            # advanced indexing on a single dimension
            selected_polygons = []
            if isinstance(item, torch.Tensor) and item.dtype == torch.bool:
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            for i in item:
                selected_polygons.append(self.polygons[i])
        return PolygonList(selected_polygons, size=self.size)

    def __iter__(self):
        return iter(self.polygons)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.polygons))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s
```

#### SegmentationMask

```python
class SegmentationMask(object):

    """
    This class stores the segmentations for all objects in the image.
    It wraps BinaryMaskList and PolygonList conveniently.
    """

    def __init__(self, instances, size, mode="poly"):
        """
        Arguments:
            instances: two types
                (1) polygon
                (2) binary mask
            size: (width, height)
            mode: 'poly', 'mask'. if mode is 'mask', convert mask of any format to binary mask
        """

        assert isinstance(size, (list, tuple))
        assert len(size) == 2
        if isinstance(size[0], torch.Tensor):
            assert isinstance(size[1], torch.Tensor)
            size = size[0].item(), size[1].item()

        assert isinstance(size[0], (int, float))
        assert isinstance(size[1], (int, float))

        if mode == "poly":
            self.instances = PolygonList(instances, size)
        elif mode == "mask":
            self.instances = BinaryMaskList(instances, size)
        else:
            raise NotImplementedError("Unknown mode: %s" % str(mode))

        self.mode = mode
        self.size = tuple(size)

    def transpose(self, method):
        flipped_instances = self.instances.transpose(method)
        return SegmentationMask(flipped_instances, self.size, self.mode)

    def crop(self, box):
        cropped_instances = self.instances.crop(box)
        cropped_size = cropped_instances.size
        return SegmentationMask(cropped_instances, cropped_size, self.mode)

    def resize(self, size, *args, **kwargs):
        resized_instances = self.instances.resize(size)
        resized_size = size
        return SegmentationMask(resized_instances, resized_size, self.mode)

    def to(self, *args, **kwargs):
        return self

    def convert(self, mode):
        if mode == self.mode:
            return self

        if mode == "poly":
            converted_instances = self.instances.convert_to_polygon()
        elif mode == "mask":
            converted_instances = self.instances.convert_to_binarymask()
        else:
            raise NotImplementedError("Unknown mode: %s" % str(mode))

        return SegmentationMask(converted_instances, self.size, mode)

    def get_mask_tensor(self):
        instances = self.instances
        if self.mode == "poly":
            instances = instances.convert_to_binarymask()
        # If there is only 1 instance
        return instances.masks.squeeze(0)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        selected_instances = self.instances.__getitem__(item)
        return SegmentationMask(selected_instances, self.size, self.mode)

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.iter_idx < self.__len__():
            next_segmentation = self.__getitem__(self.iter_idx)
            self.iter_idx += 1
            return next_segmentation
        raise StopIteration()

    next = __next__  # Python 2 compatibility

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.instances))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s
```

