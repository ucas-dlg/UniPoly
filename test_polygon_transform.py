import numpy as np
import random
import cv2
from shapely.geometry import Polygon
from shapely.affinity import rotate, scale, translate
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from datasets import build_dataset
import os
import torch


# 一个函数，接受坐标并返回变换后的坐标
def apply_affine_transform(polygon, transform):
    poly = Polygon(polygon)
    transformed_poly = transform(poly)
    return list(transformed_poly.exterior.coords)

# 随机缩放
def random_resize(image, polygon, min_scale=0.8, max_scale=1.2):
    scale = random.uniform(min_scale, max_scale)
    height, width = image.shape[:2]
    # 计算新的图像尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)
    # 使用OpenCV进行resize
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # 缩放多边形
    transform = scale(polygon, xfact=scale, yfact=scale, origin=(width/2, height/2))
    new_polygon = apply_affine_transform(polygon, transform)
    
    return resized_image, new_polygon

# 随机旋转
def random_rotate(image, polygon, max_angle=30):
    angle = random.uniform(-max_angle, max_angle)
    height, width = image.shape[:2]
    
    # 旋转矩阵
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (width, height))
    
    # 旋转多边形
    transform = rotate(polygon, angle, origin=(width/2, height/2))
    new_polygon = apply_affine_transform(polygon, transform)
    
    return rotated_image, new_polygon

# 随机裁剪
def random_crop(image, polygon, crop_height, crop_width):
    height, width = image.shape[:2]
    
    # 随机选择裁剪的起始位置
    top = random.randint(0, height - crop_height)
    left = random.randint(0, width - crop_width)
    
    cropped_image = image[top:top + crop_height, left:left + crop_width]
    
    # 裁剪多边形
    transform = translate(polygon, xoff=-left, yoff=-top)
    new_polygon = apply_affine_transform(transform, polygon)
    
    return cropped_image, new_polygon

# 示例：对图像及其分割多边形进行增强
def apply_augmentation(image, segmentation):
    # 选择变换
    image, segmentation = random_resize(image, segmentation)
    image, segmentation = random_rotate(image, segmentation)
    image, segmentation = random_crop(image, segmentation, 200, 200)
    
    return image, segmentation


def draw_polygons(image, polygons, color_palette=(0, 255, 0), thickness=2):
    for polygon in polygons:
        color = random.choice(color_palette)
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness)
    return image

def draw_bbox(image, bboxes, color=(255, 0, 0), thickness=2):
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1 - x2//2, y1 - y2//2), (x1 + x2//2, y1 + y2//2), color, thickness)
    return image

def draw_bbox_xyxy(image, bboxes, color=(255, 0, 0), thickness=2):
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1 , y1), (x2, y2), color, thickness)
    return image

def main():
    color_palette = [
    (255, 255, 0),  # Bright Yellow
    (0, 255, 0),  # Bright Green
    (255, 0, 0),  # Bright Red
    (255, 165, 0),  # Orange
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (173, 255, 47),  # Green-Yellow
    (255, 105, 180),  # Hot Pink
    (127, 255, 0),  # Chartreuse
    (255, 69, 0),  # Red-Orange
    ]
    import argparse
    parser = argparse.ArgumentParser('Test Polygon Transform', add_help=False)
    parser.add_argument('--dataset_file', default='road_instance_pts_v1')  # all_pts building_instance_pts
    parser.add_argument('--coco_path', default='/irsa/ROAD_DATA/sat2graph_sn3_dataset/RGB_1.0_meter', type=str)
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--output_dir', default=None, help='path where to save, empty for no saving')
    args = parser.parse_args()
    vis_path = '/irsa/build_code/Deformable-DETR/weights/vis'
    os.makedirs(vis_path, exist_ok=True) 
    dataset_train = build_dataset(image_set='train', args=args)
    for i, (image, target) in enumerate(dataset_train):
        
        # image = np.array(image)
        # cv2.imwrite(os.path.join(vis_path, f'{i}.png'), image)
        bbox = target['boxes'] * (target['size'].repeat(2))
        bbox = np.array(bbox, dtype=np.int32)

        image = draw_bbox(np.array(image), bbox, color=(255, 0, 0), thickness=1)
        # image = draw_bbox_xyxy(np.array(image), bbox, color=(255, 0, 0), thickness=1)
        # instances_pts = target['instances_pts'].numpy()
        # instances_pts = instances_pts.reshape(instances_pts.shape[0] ,-1, 2)
        instances_pts = target['instances_pts'] * target['size']
        image = draw_polygons(np.array(image), instances_pts, color_palette, thickness=2)
        cv2.imwrite(os.path.join(vis_path, f'{i}.png'), image)
        print(i)


if __name__ == "__main__":

    main()