# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate

from shapely.geometry import LineString, box, Polygon
from shapely import affinity
import math
import numpy as np

def remove_padding(segmentation, pad_value=-1):
    """
    去掉 segmentation 中所有等于 pad_value 的元素（默认 -1）。
    """
    return segmentation[segmentation != pad_value]



def tensor_to_linestring(segmentation: torch.Tensor):
    """
    将一维 tensor 转换为 shapely.geometry.LineString
    """
    # 转换为 numpy 数组
    coords = segmentation.view(-1, 2).numpy()
    # 构造 LineString
    line = LineString(coords)
    return line

def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target

def crop_v2(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    instance_pts = target['instances_pts']
    crop_linestrings = []
    for linestring_pts_ in instance_pts:
        if len(set(list(linestring_pts_.coords))) < 3: continue
        linestring_pts = Polygon(list(linestring_pts_.coords)).buffer(0)
        cropped_linestring = linestring_pts.intersection(box(j, i, j + w, i + h))
        # 将裁剪后的几何图形平移到新的坐标系中
        translated_linestring = affinity.translate(cropped_linestring, xoff=-j, yoff=-i)
        if translated_linestring.geom_type == 'LineString':
            crop_linestrings.append(translated_linestring)
        elif translated_linestring.geom_type == 'Polygon' and translated_linestring.is_empty == False:
            crop_linestrings.append(LineString(translated_linestring.exterior))
        elif translated_linestring.geom_type == 'Point':
            continue
        elif translated_linestring.geom_type == 'MultiLineString' or translated_linestring.geom_type == 'MultiPolygon':
            for geom in translated_linestring.geoms:
                if geom.geom_type == 'LineString':
                    crop_linestrings.append(geom)
                elif geom.geom_type == 'Polygon' and geom.is_empty == False:
                    crop_linestrings.append(LineString(geom.exterior))
    target['instances_pts'] = crop_linestrings

    return cropped_image, target

def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def horizontal_flip(image, target):
    # 水平翻转
    flipped_image = F.hflip(image)

    w, h = image.size
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "instances_pts" in target:
        instance_pts = target['instances_pts']
        instance_linestrings = []
        for i in range(instance_pts.shape[0]):
            valid_pts = remove_padding(instance_pts[i])
            linestring_pts = tensor_to_linestring(valid_pts)
            hflip_linestring = affinity.scale(linestring_pts, xfact=-1, yfact=1, origin=((w - 1) / 2.0, 0))
            instance_linestrings.append(hflip_linestring)
        target['instances_pts'] = instance_linestrings

    return flipped_image, target


def vertical_flip(image, target):
    # 垂直翻转
    flipped_image = F.vflip(image)
    
    w, h = image.size
    target = target.copy()
    
    # 翻转 bounding boxes
    if "boxes" in target:
        boxes = target["boxes"]
        # y1, y2 互换，并根据高度进行翻转
        boxes = boxes[:, [0, 3, 2, 1]] * torch.as_tensor([1, -1, 1, -1]) + torch.as_tensor([0, h, 0, h])
        target["boxes"] = boxes
    
    # 翻转实例点（折线或多边形）
    if "instances_pts" in target:
        instance_pts = target['instances_pts']
        instance_linestrings = []
        for i in range(instance_pts.shape[0]):
            valid_pts = remove_padding(instance_pts[i])
            linestring_pts = tensor_to_linestring(valid_pts)
            vflip_linestring = affinity.scale(linestring_pts, xfact=1, yfact=-1, origin=(0, (h - 1) / 2.0))
            instance_linestrings.append(vflip_linestring)
        target['instances_pts'] = instance_linestrings
    
    return flipped_image, target

def diagonal_flip(image, target):
    # 对角线翻转
    flipped_image = F.hflip(image)
    flipped_image = F.vflip(flipped_image)
    
    w, h = image.size
    target = target.copy()

    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes * torch.as_tensor([-1, -1, -1, -1]) + torch.as_tensor([w, h, w, h])
        target["boxes"] = boxes

    if "instances_pts" in target:
        instance_pts = target['instances_pts']
        instance_linestrings = []
        for i in range(instance_pts.shape[0]):
            valid_pts = remove_padding(instance_pts[i])
            linestring_pts = tensor_to_linestring(valid_pts)
            # 对角线翻转：x和y轴都翻转，原点为(w-1, h-1)
            diag_flip_linestring = affinity.scale(linestring_pts, xfact=-1, yfact=-1, origin=((w - 1) / 2.0, (h - 1) / 2.0))
            instance_linestrings.append(diag_flip_linestring)
        target['instances_pts'] = instance_linestrings
    
    return flipped_image, target

def rotate(image, target, angle):
    rotated_image = F.rotate(image, -angle)

    w, h = image.size
    w = w - 1
    h = h - 1
    target = target.copy()

    if "boxes" in target:
        boxes = target["boxes"]
        # 旋转90度
        if angle == 90:
            # (x1, y1, x2, y2) -> (h - y2, x1, h - y1, x2)
            # boxes = torch.stack([boxes[:, 1], w - boxes[:, 2], boxes[:, 3], w - boxes[:, 0]], dim=1)
            boxes = torch.stack([h - boxes[:, 3], boxes[:, 0], h - boxes[:, 1], boxes[:, 2]], dim=1)
            # boxes = boxes[:, [1, 0, 3, 2]]
            # boxes = boxes * torch.as_tensor([-1, -1, -1, -1]) + torch.as_tensor([h, w, h, w])
        elif angle == 180:
            # (x1, y1, x2, y2) -> (w - x2, h - y2, w - x1, h - y1)
            boxes = boxes[:, [2, 3, 0, 1]]
            boxes = boxes * torch.as_tensor([-1, -1, -1, -1]) + torch.as_tensor([w, h, w, h])
        elif angle == 270:
            # (x1, y1, x2, y2) -> (y1, w - x2, y2, w - x1)
            boxes = torch.stack([boxes[:, 1], w - boxes[:, 2], boxes[:, 3], w - boxes[:, 0]], dim=1)
        target["boxes"] = boxes

    if "instances_pts" in target:
        instance_pts = target['instances_pts']
        instance_linestrings = []
        for i in range(len(instance_pts)):
            linestring_pts = instance_pts[i]
            origin = (w/2.0, h/2.0)
            if angle == 90:
                # 对点进行90度旋转
                rotated_linestring = affinity.rotate(linestring_pts, 90, origin=origin)
            elif angle == 180:
                # 对点进行180度旋转
                rotated_linestring = affinity.rotate(linestring_pts, 180, origin=origin)
            elif angle == 270:
                # 对点进行270度旋转
                rotated_linestring = affinity.rotate(linestring_pts, 270, origin=origin)
            
            instance_linestrings.append(rotated_linestring)
        target['instances_pts'] = instance_linestrings

    return rotated_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
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

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    # if target['orig_size'][0] == 512:
    #     print(target['orig_size'], target['size'])  # DEBUG

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target

def resize_v2(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
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

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "instances_pts" in target:
        instance_pts = target['instances_pts']
        scaled_instance_pts = instance_pts * torch.as_tensor([ratio_width, ratio_height])
        target['instances_pts'] = scaled_instance_pts

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def resize_v3(image, target, size, max_size=None):  # bbox改成4个坐标  旋转框的形式
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
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

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height] * (boxes.shape[-1] // 2))
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "instances_pts" in target:
        instance_pts = target['instances_pts']
        scaled_instance_pts = instance_pts * torch.as_tensor([ratio_width, ratio_height])
        target['instances_pts'] = scaled_instance_pts

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target

def resize_v4(image, target, size, max_size=None):  # bbox改成4个坐标  旋转框的形式
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
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

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes[:, :4] = boxes[:, :4] * torch.tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "instances_pts" in target:
        instance_pts = target['instances_pts']
        scaled_instance_pts = instance_pts * torch.as_tensor([ratio_width, ratio_height])
        target['instances_pts'] = scaled_instance_pts

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target

def resize_v5(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
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

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "instances_pts" in target:
        instance_pts = target['instances_pts']
        scaled_instance_pts = [LineString([(x * ratio_width, y * ratio_height) for x, y in line.coords]) for line in instance_pts]
        target['instances_pts'] = scaled_instance_pts

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target

def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target

def pad_v2(image, target, padding):  # add by DLG
    if image.size[0] >= padding[0] and image.size[1] >= padding[1]:
        return image, target
    padded_image = F.pad(image, (0, 0, padding[0] - image.size[0], padding[1] - image.size[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size)
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target

def interpolate_to_points_with_classes(coordinates, target_points=64):
    """插值到 指定数量点，同时生成类别"""
    # 将点对格式化为 [(x1, y1), (x2, y2), ...]
    points = coordinates
    original_point_count = len(points)
    total_new_points = target_points - original_point_count  # 需要插入的点数

    if total_new_points <= 0:
        return [coord for point in points for coord in point], [1] * original_point_count  # 不需要插值，直接返回原点和类别

    # 计算每段的欧式距离
    distances = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        distances.append(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    # 每段距离占总距离的比例，用于分配插值点数
    total_distance = sum(distances)
    segment_ratios = [d / total_distance for d in distances]

    # 为每段分配插值点数（尽量均匀）
    segment_new_points = [math.floor(r * total_new_points) for r in segment_ratios]
    remaining_points = total_new_points - sum(segment_new_points)

    # 分配剩余的插值点到段
    for i in range(remaining_points):
        segment_new_points[i % len(segment_new_points)] += 1

    # 开始插值并生成类别
    new_points = []
    point_classes = []  # 类别列表
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        # 添加当前段的起始点，并设置类别为1（原始点）
        new_points.append((x1, y1))
        point_classes.append(1)

        # 均匀插值点
        num_new_points = segment_new_points[i]
        for j in range(1, num_new_points + 1):
            t = j / (num_new_points + 1)
            new_x = x1 + t * (x2 - x1)
            new_y = y1 + t * (y2 - y1)
            new_points.append((new_x, new_y))
            point_classes.append(0)  # 插值点类别为2

    # 添加最后一个点，并设置类别为1（原始点）
    for i in range(target_points - len(new_points)):
        new_points.append(points[-1])
        point_classes.append(1)

    # 将点展平为 [x1, y1, x2, y2, ...] 的格式
    result_coordinates = [coord for point in new_points for coord in point]
    return result_coordinates, point_classes


def is_clockwise(pts):
    """
    判断闭合曲线是顺时针还是逆时针
    :param pts: 包含点坐标的列表 [(x1, y1), (x2, y2), ..., (xn, yn)]
    :return: True 表示顺时针，False 表示逆时针
    """
    return sum((pts[i+1][0] - pts[i][0]) * (pts[i+1][1] + pts[i][1]) for i in range(len(pts) - 1)) + \
           (pts[-1][0] - pts[0][0]) * (pts[-1][1] + pts[0][1]) < 0


def set_start_point(points, y_tol: float = 2.0):
    """
    将 polygon 点序列的起点设为：
      1) y 最小的点；
      2) 若有多个点的 y 与最小 y 的差 <= y_tol(像素)，从这些候选中取 x 最小者；
    然后对序列做循环旋转，使该点成为第一个点，并保持原有相对顺序不变。
    
    兼容闭合多边形（首尾点相同）；返回结果与输入的闭合性一致。
    """
    if not points:
        return points[:]

    # 检测是否闭合（首尾相同）
    is_closed = len(points) > 1 and points[0] == points[-1]
    core = points[:-1] if is_closed else points[:]  # 去掉末尾重复点再处理

    # 1) 找全局最小 y
    min_y = min(p[1] for p in core)

    # 2) 在 y 与 min_y 的差 <= y_tol 的候选中选 x 最小的点
    candidates = [(i, p) for i, p in enumerate(core) if (p[1] - min_y) <= y_tol]
    # 按 x、再按原始索引稳定选择（若 x 相同，取原序靠前的）
    start_idx = min(candidates, key=lambda t: (t[1][0], t[0]))[0]

    # 3) 旋转序列：保持相对顺序不变，只改变起点位置
    rotated = core[start_idx:] + core[:start_idx]

    # 若输入是闭合多边形，则在末尾补回首点
    if is_closed:
        rotated.append(rotated[0])

    return rotated


def reorder_linestring(linestring_pts, y_tolerance=2):
    """
    如果 linestring_pts.coords 的最后一个点比第零个点 y 坐标小，
    或者 y 坐标相同时 x 坐标更小，则将点序倒序排列。
    允许 y 坐标相差 y_tolerance 像素以内视为相等。
    """
    coords = list(linestring_pts.coords)
    first = coords[0]
    last = coords[-1]

    y_diff = abs(last[1] - first[1])
    if y_diff <= y_tolerance:
        # y 坐标视为相等，比 x
        if last[0] < first[0]:
            coords.reverse()
    elif last[1] < first[1]:
        # y 坐标更小，反转
        coords.reverse()

    return LineString(coords)

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)
    
class RandomCropV2(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        if img.size[0] <= self.size[0] or img.size[1] <= self.size[1]:
            return pad_v2(img, target, self.size)
        region = T.RandomCrop.get_params(img, self.size)
        return crop_v2(img, target, region)
    
class PadV2(object):  # add by DLG
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img, target):
        return pad_v2(img, target, self.size)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target

class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return random.choice([horizontal_flip, vertical_flip, diagonal_flip])(img, target)
        else:
            if "instances_pts" in target:
                target = target.copy()
                instance_pts = target['instances_pts']
                instance_linestrings = []
                for i in range(instance_pts.shape[0]):
                    valid_pts = remove_padding(instance_pts[i])
                    linestring_pts = tensor_to_linestring(valid_pts)
                    instance_linestrings.append(linestring_pts)
                target['instances_pts'] = instance_linestrings
            return img, target

class RandomRotate(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            angle = random.choice([90, 180, 270])
            # angle = 270
            img, target = rotate(img, target, angle)

            return img, target
        return img, target

class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)
    
class RandomResize_v2(object):  # 添加instance_pts字段
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize_v2(img, target, size, self.max_size)
    
class RandomResize_v3(object):  # 添加instance_pts字段
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize_v3(img, target, size, self.max_size)
    
class RandomResize_v4(object):  # 添加instance_pts字段
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize_v4(img, target, size, self.max_size)
    
    
class RandomResize_v5(object):  # 添加instance_pts字段
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize_v5(img, target, size, self.max_size)

class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target

class NormalizeV2(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        # image = np.array(image)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        # h, w = image.shape[0:2]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        if "instances_pts" in target:
            instance_pts = target["instances_pts"]
            instance_pts = instance_pts / torch.tensor([w, h], dtype=torch.float32)
            target["instances_pts"] = instance_pts
        return image, target
    
class NormalizeV3(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        # h, w = image.size
        if "boxes" in target:
            boxes = target["boxes"]
            # boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h] * (boxes.shape[-1] // 2), dtype=torch.float32)
            target["boxes"] = boxes
        if "instances_pts" in target:
            instance_pts = target["instances_pts"]
            instance_pts = instance_pts / torch.tensor([w, h], dtype=torch.float32)
            target["instances_pts"] = instance_pts
        return image, target

class NormalizeV4(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        # h, w = image.size
        if "boxes" in target:
            boxes = target["boxes"]
            # boxes = box_xyxy_to_cxcywh(boxes)
            boxes[:, :4] = boxes[:, :4] / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        if "instances_pts" in target:
            instance_pts = target["instances_pts"]
            instance_pts = instance_pts / torch.tensor([w, h], dtype=torch.float32)
            target["instances_pts"] = instance_pts
        return image, target

class linestring2pt(object):
    def __call__(self, image, target):
        target = target.copy()
        instances_pts = target['instances_pts']
        new_pts = []
        new_pts_classes = []
        new_bboxes = []
        new_bboxes_area = []
        for i in range(len(instances_pts)):
            linestring_pts = instances_pts[i]
            if len(set(list(linestring_pts.coords))) < 3 or len(linestring_pts.coords) > 64: continue  # 过滤掉过短或过长的线段
            if linestring_pts.is_ring:
                if not is_clockwise(linestring_pts.coords):
                    linestring_pts = LineString(linestring_pts.coords[::-1])
                temp_polygon = Polygon(linestring_pts)
                reorder_linestring_pts = LineString(set_start_point(linestring_pts.coords, y_tol=2))  # 固定起点和方向
            else:
                linestring_pts = LineString(list(linestring_pts.coords) + [linestring_pts.coords[0]])  # 闭合
                if not is_clockwise(linestring_pts.coords):
                    linestring_pts = LineString(linestring_pts.coords[::-1])
                temp_polygon = Polygon(linestring_pts)
                reorder_linestring_pts = LineString(set_start_point(linestring_pts.coords, y_tol=2))  # 固定起点和方向
            if temp_polygon.area < 9 or reorder_linestring_pts.bounds[2]-reorder_linestring_pts.bounds[0]<3 or reorder_linestring_pts.bounds[3]-reorder_linestring_pts.bounds[1]<3: 
                continue
            # 插值到64个点
            pts, classes = interpolate_to_points_with_classes(reorder_linestring_pts.coords, target_points=64) 
            new_pts.append(pts)
            new_pts_classes.append(classes)
            new_bboxes.append(reorder_linestring_pts.bounds)
            new_bboxes_area.append(reorder_linestring_pts.area)
        if len(new_pts) == 0:
            target['instances_pts'] = torch.zeros((0, 64, 2), dtype=torch.float32)
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.ones((len(new_pts),), dtype=torch.int64)
            target['area'] = torch.zeros((0,), dtype=torch.float32)
            target['coord_class'] = torch.zeros((0, 64), dtype=torch.float32)
            return image, target
        # tensor_pts = torch.tensor(new_pts, dtype=torch.float32)
        # if tensor_pts.shape[1] != 128:
        #     print(tensor_pts.shape)
        target['instances_pts'] = torch.tensor(new_pts, dtype=torch.float32).reshape(len(new_pts), -1, 2)
        target['boxes'] = torch.tensor(new_bboxes, dtype=torch.float32)
        target['labels'] = torch.ones((len(new_pts),), dtype=torch.int64)
        target['area'] = torch.tensor(new_bboxes_area, dtype=torch.float32)
        target['coord_class'] = torch.tensor(new_pts_classes, dtype=torch.float32)
        return image, target

class linestring2ptV2(object): # 之前是顺时针，现在改成逆时针
    def __call__(self, image, target):
        target = target.copy()
        instances_pts = target['instances_pts']
        new_pts = []
        new_pts_classes = []
        new_bboxes = []
        new_bboxes_area = []
        for i in range(len(instances_pts)):
            linestring_pts = instances_pts[i]
            if len(set(list(linestring_pts.coords))) < 3 or len(linestring_pts.coords) > 64: continue  # 过滤掉过短或过长的线段
            if linestring_pts.is_ring:
                if is_clockwise(linestring_pts.coords):
                    linestring_pts = LineString(linestring_pts.coords[::-1])
                temp_polygon = Polygon(linestring_pts)
                reorder_linestring_pts = LineString(set_start_point(linestring_pts.coords, y_tol=2))  # 固定起点和方向
            else:
                linestring_pts = LineString(list(linestring_pts.coords) + [linestring_pts.coords[0]])  # 闭合
                if is_clockwise(linestring_pts.coords):
                    linestring_pts = LineString(linestring_pts.coords[::-1])
                temp_polygon = Polygon(linestring_pts)
                reorder_linestring_pts = LineString(set_start_point(linestring_pts.coords, y_tol=2))  # 固定起点和方向
            if (reorder_linestring_pts.bounds[3] - reorder_linestring_pts.bounds[1])  * (reorder_linestring_pts.bounds[2] - reorder_linestring_pts.bounds[0]) < 9 or reorder_linestring_pts.bounds[2]-reorder_linestring_pts.bounds[0]<3 or reorder_linestring_pts.bounds[3]-reorder_linestring_pts.bounds[1]<3: 
                continue
            # 插值到64个点
            pts, classes = interpolate_to_points_with_classes(reorder_linestring_pts.coords, target_points=64) 
            new_pts.append(pts)
            new_pts_classes.append(classes)
            new_bboxes.append(reorder_linestring_pts.bounds)
            new_bboxes_area.append((reorder_linestring_pts.bounds[3] - reorder_linestring_pts.bounds[1])  * (reorder_linestring_pts.bounds[2] - reorder_linestring_pts.bounds[0]))
        if len(new_pts) == 0:
            target['instances_pts'] = torch.zeros((0, 64, 2), dtype=torch.float32)
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.ones((len(new_pts),), dtype=torch.int64)
            target['area'] = torch.zeros((0,), dtype=torch.float32)
            target['coord_class'] = torch.zeros((0, 64), dtype=torch.float32)

            # target['instances_pts'] = torch.zeros((1, 64, 2), dtype=torch.float32)
            # target['boxes'] = torch.zeros((1, 4), dtype=torch.float32)
            # target['labels'] = torch.ones((len(new_pts) + 1,), dtype=torch.int64)
            # target['area'] = torch.zeros((1,), dtype=torch.float32)
            # target['coord_class'] = torch.zeros((1, 64), dtype=torch.float32)
        
            return image, target
        # tensor_pts = torch.tensor(new_pts, dtype=torch.float32)
        # if tensor_pts.shape[1] != 128:
        #     print(tensor_pts.shape)
        target['instances_pts'] = torch.tensor(new_pts, dtype=torch.float32).reshape(len(new_pts), -1, 2)
        target['boxes'] = torch.tensor(new_bboxes, dtype=torch.float32)
        target['labels'] = torch.ones((len(new_pts),), dtype=torch.int64)
        target['area'] = torch.tensor(new_bboxes_area, dtype=torch.float32)
        target['coord_class'] = torch.tensor(new_pts_classes, dtype=torch.float32)
        return image, target

class linestring2ptV3(object):
    #  相比于v2 建筑物类别数改成了3  （背景类别0，非闭合线段类别1，闭合线段类别2）
    def __call__(self, image, target):
        target = target.copy()
        instances_pts = target['instances_pts']
        new_pts = []
        new_pts_classes = []
        new_bboxes = []
        new_bboxes_area = []
        for i in range(len(instances_pts)):
            linestring_pts = instances_pts[i]
            if len(set(list(linestring_pts.coords))) < 3 or len(linestring_pts.coords) > 64: continue  # 过滤掉过短或过长的线段
            if linestring_pts.is_ring:
                if is_clockwise(linestring_pts.coords):
                    linestring_pts = LineString(linestring_pts.coords[::-1])
                temp_polygon = Polygon(linestring_pts)
                reorder_linestring_pts = LineString(set_start_point(linestring_pts.coords, y_tol=2))  # 固定起点和方向
            else:
                linestring_pts = LineString(list(linestring_pts.coords) + [linestring_pts.coords[0]])  # 闭合
                if is_clockwise(linestring_pts.coords):
                    linestring_pts = LineString(linestring_pts.coords[::-1])
                temp_polygon = Polygon(linestring_pts)
                reorder_linestring_pts = LineString(set_start_point(linestring_pts.coords, y_tol=2))  # 固定起点和方向
            if (reorder_linestring_pts.bounds[3] - reorder_linestring_pts.bounds[1])  * (reorder_linestring_pts.bounds[2] - reorder_linestring_pts.bounds[0]) < 9 or reorder_linestring_pts.bounds[2]-reorder_linestring_pts.bounds[0]<3 or reorder_linestring_pts.bounds[3]-reorder_linestring_pts.bounds[1]<3: 
                continue
            # 插值到64个点
            pts, classes = interpolate_to_points_with_classes(reorder_linestring_pts.coords, target_points=64) 
            new_pts.append(pts)
            new_pts_classes.append(classes)
            new_bboxes.append(reorder_linestring_pts.bounds)
            new_bboxes_area.append((reorder_linestring_pts.bounds[3] - reorder_linestring_pts.bounds[1])  * (reorder_linestring_pts.bounds[2] - reorder_linestring_pts.bounds[0]))
        if len(new_pts) == 0:
            target['instances_pts'] = torch.zeros((0, 64, 2), dtype=torch.float32)
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.ones((len(new_pts),), dtype=torch.int64)
            target['area'] = torch.zeros((0,), dtype=torch.float32)
            target['coord_class'] = torch.zeros((0, 64), dtype=torch.float32)

            # target['instances_pts'] = torch.zeros((1, 64, 2), dtype=torch.float32)
            # target['boxes'] = torch.zeros((1, 4), dtype=torch.float32)
            # target['labels'] = torch.ones((len(new_pts) + 1,), dtype=torch.int64)
            # target['area'] = torch.zeros((1,), dtype=torch.float32)
            # target['coord_class'] = torch.zeros((1, 64), dtype=torch.float32)
        
            return image, target
        # tensor_pts = torch.tensor(new_pts, dtype=torch.float32)
        # if tensor_pts.shape[1] != 128:
        #     print(tensor_pts.shape)
        temp_pts = torch.tensor(new_pts, dtype=torch.float32).reshape(len(new_pts), -1, 2)
        target['instances_pts'] = temp_pts
        target['boxes'] = torch.tensor(new_bboxes, dtype=torch.float32)
        target['labels'] = ((temp_pts[:, 0] == temp_pts[:, -1]).all(dim=1).int() + 1).to(torch.int64)  # 闭合为2类，非闭合为1类
        target['area'] = torch.tensor(new_bboxes_area, dtype=torch.float32)
        target['coord_class'] = torch.tensor(new_pts_classes, dtype=torch.float32)
        return image, target

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
