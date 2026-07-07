# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco

from .all_pts import build as build_all_pts

from .instance_pts import build as build_instance_pts


from .road_instance_pts_v6 import build as build_road_instance_pts_v6

from .road_point_bbox import build as build_road_point_bbox

from .topoboundary_instance_pts_v4 import build as build_topoboundary_instance_pts_v4  # topo boundary 数据集


from .crowdai_instance_pts_v2 import build as build_crowdai_instance_pts_v2

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'all_pts':
        return build_all_pts(image_set, args)
    if args.dataset_file == 'instance_pts':
        return build_instance_pts(image_set, args)

    if args.dataset_file == 'road_instance_pts_v6':
        return build_road_instance_pts_v6(image_set, args)
    if args.dataset_file == 'road_point_bbox':
        return build_road_point_bbox(image_set, args)
    if args.dataset_file == 'topoboundary_instance_pts_v4':
        return build_topoboundary_instance_pts_v4(image_set, args)
    if args.dataset_file == 'crowdai_instance_pts_v2':
        return build_crowdai_instance_pts_v2(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')
