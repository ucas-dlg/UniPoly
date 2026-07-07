# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.dpbuilding_eval import BuildingEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
import time
import pickle
import numpy as np
import json
import subprocess
from tqdm import tqdm
import cv2
from scipy.spatial import cKDTree
from PIL import Image
from skimage import measure
import shutil
import torch.distributed as dist
from pycocotools.coco import COCO
from pycocotools import mask as cocomask




def build_undirected_graph(points_px: np.ndarray, wrap_threshold: float = 5.0):
    """
    points_px: numpy 数组，形状 (N, 2)，单位：像素（注意你画线时 *1000 了）
    wrap_threshold: 首尾两点作为相邻点的欧式距离阈值（像素）
    返回：{ (x, y): [(nx1, ny1), (nx2, ny2), ...], ... }
    """
    pts = np.asarray(points_px, dtype=np.float32)
    N = pts.shape[0]
    if N < 2:
        raise ValueError("points length must be >= 2")

    coords = [tuple(map(float, p)) for p in pts]  # key 用 (x, y) tuple

    # 首尾是否闭合
    wrap = np.linalg.norm(pts[0] - pts[-1]) <= wrap_threshold

    graph = {}
    for i in range(N):
        neighbors_idx = []
        if i - 1 >= 0:
            neighbors_idx.append(i - 1)
        if i + 1 < N:
            neighbors_idx.append(i + 1)
        if wrap:
            if i == 0:
                neighbors_idx.append(N - 1)
            elif i == N - 1:
                neighbors_idx.append(0)

        # 去重并按顺序
        seen = set()
        ordered_neighbors = []
        for j in neighbors_idx:
            if j not in seen:
                ordered_neighbors.append(coords[j])
                seen.add(j)

        graph[coords[i]] = ordered_neighbors

    return graph

def build_undirected_graph_v2(points_px: np.ndarray, is_ring):
    """
    points_px: numpy 数组，形状 (N, 2)，单位：像素（注意你画线时 *1000 了）
    is_ring: 是否收尾相连
    返回：{ (x, y): [(nx1, ny1), (nx2, ny2), ...], ... }
    """
    pts = np.asarray(points_px, dtype=np.float32)
    N = pts.shape[0]
    if N < 2:
        raise ValueError("points length must be >= 2")

    coords = [tuple(map(float, p)) for p in pts]  # key 用 (x, y) tuple


    graph = {}
    for i in range(N):
        neighbors_idx = []
        if i - 1 >= 0:
            neighbors_idx.append(i - 1)
        if i + 1 < N:
            neighbors_idx.append(i + 1)
        if is_ring:
            if i == 0:
                neighbors_idx.append(N - 1)
            elif i == N - 1:
                neighbors_idx.append(0)

        # 去重并按顺序
        seen = set()
        ordered_neighbors = []
        for j in neighbors_idx:
            if j not in seen:
                ordered_neighbors.append(coords[j])
                seen.add(j)

        graph[coords[i]] = ordered_neighbors

    return graph

def merge_graph_dicts(dict1, dict2):
    for k, v in dict2.items():
        if k in dict1:
            dict1[k] = list(set(dict1[k] + v))
        else:
            dict1[k] = v
    return dict1

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100 # 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    torch.cuda.empty_cache()  # 视需要：每 N 次再清一次更合理
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        t1 = time.time()
        outputs = model(samples)
        t2 = time.time()
        # print("model time: ", t2 - t1)
        loss_dict = criterion(outputs, targets)
        t3 = time.time()
        # print("criterion time: ", t3 - t2)
        weight_dict = criterion.weight_dict
        # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        total_loss = 0

        # 遍历 loss_dict 中的每个键
        for k in loss_dict.keys():
            # 检查当前键是否在 weight_dict 中
            if k in weight_dict:
                # 计算当前键对应的损失并加权
                weighted_loss = loss_dict[k] * weight_dict[k]
                # 累加到总损失
                total_loss += weighted_loss

        # 最终的损失值
        losses = total_loss

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        t4 = time.time()
        # print("backward+opt time: ", t4 - t3)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        # total_memory = torch.cuda.get_device_properties(0).total_memory
        # allocated_memory = torch.cuda.memory_allocated(0)
        # reserved_memory = torch.cuda.memory_reserved(0)
    
        # # 计算当前显存使用比例
        # memory_usage = reserved_memory / total_memory
        # if memory_usage > 0.95:  # 如果使用比例超过90%
        # if _ % 100 == 0:  # 每10个batch清理一次
        # # # del outputs, samples, targets, loss_dict
        #     torch.cuda.empty_cache()  # 视需要：每 N 次再清一次更合理

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    torch.cuda.empty_cache()  # 视需要：每 N 次再清一次更合理
    # for samples, targets in metric_logger.log_every(data_loader, 10, header):
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # =============== 注释掉计算损失的部分 ================= #
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        # ================================================== #

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        del samples, targets, outputs

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    return coco_evaluator
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # if coco_evaluator is not None:
    #     if 'bbox' in postprocessors.keys():
    #         stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    #     if 'segm' in postprocessors.keys():
    #         stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]
    # return stats, coco_evaluator

@torch.no_grad()
def evaluate_building(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = ('segm', )  # 只评估segm
    coco_evaluator = BuildingEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    torch.cuda.empty_cache()  # 视需要：每 N 次再清一次更合理
    # for samples, targets in metric_logger.log_every(data_loader, 10, header):
    with torch.inference_mode(): 
        for samples, targets in data_loader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            # results = postprocessors['bbox'](outputs, orig_target_sizes)
            if 'segm' in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors['segm'](outputs, orig_target_sizes, target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            if coco_evaluator is not None:
                coco_evaluator.update(res)

            del samples, targets, outputs

        # gather the stats from all processes
        # metric_logger.synchronize_between_processes()
        # print("Averaged stats:", metric_logger)
        if coco_evaluator is not None:
            coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        if coco_evaluator is not None:
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
    return coco_evaluator


# ====================== 评估ldpoly =================== #
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def calc_iou(a, b):
    """Compute standard IoU between two binary masks."""
    i = np.logical_and(a, b)  # intersection
    u = np.logical_or(a, b)  # union
    I = np.sum(i)
    U = np.sum(u)

    iou = I/(U + 1e-9)

    is_void = U == 0
    if is_void:
        return 1.0
    else:
        return iou


def compute_n_ratio(pred_vertices, gt_vertices):
    """Compute vertex efficiency ratio N_pred / N_gt with edge cases handled."""
    # Handle cases where the ground truth has no vertices
    if gt_vertices == 0:
        if pred_vertices == 0:
            return 1  # Both are zero, return a ratio of 1
        else:
            return float('NaN')  # If no ground truth vertices but there are predicted, return NaN
    else:
        return pred_vertices / gt_vertices  # Normal case, compute ratio

def _decode_coco_mask(annotation, height, width):
    """Decode COCO polygon segmentation to a single binary mask and vertex count."""
    rle = cocomask.frPyObjects(annotation["segmentation"], height, width)
    m = cocomask.decode(rle)

    # Handle exterior + interior holes
    if m.ndim > 2:
        final_mask = m[:, :, 0].copy()
        for i in range(1, m.shape[-1]):
            final_mask = np.logical_and(final_mask, np.logical_not(m[:, :, i]))
        final_mask = final_mask.astype(np.uint8)
    else:
        final_mask = m.astype(np.uint8)

    num_vertices = len(annotation["segmentation"][0]) // 2
    return final_mask.reshape(height, width), num_vertices

@torch.no_grad()
def evaluate_ldpoly(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, score_threshold=0.1):
    model.eval()
    criterion.eval()

    # json_data = json.load(open('/irsa/build_code/LDPoly/data/deventer_road/annotations/test.json', 'r'))
    # id_to_filename = {img['id']: img['file_name'] for img in json_data['images']}  # 映射id和文件名对应
    gti_annotations = '/irsa/build_code/LDPoly/data/deventer_road/annotations/test.json'
    coco_gt = COCO(gti_annotations)
    torch.cuda.empty_cache()  # 视需要：每 N 次再清一次更合理
    predictions = []

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        outputs_score = outputs['pred_logits'].sigmoid()[:, :, 2]
        outputs_seg = outputs['pred_instances_pts'] * 256
        outputs_bbox = outputs['pred_boxes'] * 256

        outputs_score = outputs_score.cpu().detach().numpy()
        outputs_seg = outputs_seg.cpu().detach().numpy()
        outputs_bbox = outputs_bbox.cpu().detach().numpy()

        for batch_idx, target in enumerate(targets):
            image_id = int(target['image_id'].cpu().item())

            for score, seg, bbox in zip(outputs_score[batch_idx], outputs_seg[batch_idx], outputs_bbox[batch_idx]):
                if score <= score_threshold:
                    continue

                seg = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
                bbox = np.asarray(bbox, dtype=np.float32)

                # Deformable DETR boxes are normalized cx, cy, w, h; COCO result json uses x, y, w, h.
                cx, cy, w, h = bbox.tolist()
                x = cx - 0.5 * w
                y = cy - 0.5 * h

                predictions.append({
                    "image_id": image_id,
                    "category_id": 100,
                    "segmentation": [seg.reshape(-1).tolist()],
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "score": float(score),
                })

        del samples, targets, outputs

    if utils.is_dist_avail_and_initialized():
        all_predictions = [None for _ in range(utils.get_world_size())]
        dist.all_gather_object(all_predictions, predictions)
        predictions = []
        for rank_predictions in all_predictions:
            predictions.extend(rank_predictions)

    # if output_dir and utils.is_main_process():
    #     os.makedirs(output_dir, exist_ok=True)
    #     output_file = os.path.join(output_dir, 'ldpoly_predictions.json')
    #     with open(output_file, 'w') as f:
    #         json.dump(predictions, f)
    if len(predictions) == 0:
        return 0, 0, 0, 0, 0
    coco_dt = COCO(gti_annotations)
    coco_dt = coco_dt.loadRes(predictions)

    all_image_ids = coco_gt.getImgIds()
    pred_image_ids = set(coco_dt.getImgIds(catIds=coco_dt.getCatIds()))

    # Sanity check for ID consistency
    if not pred_image_ids.issubset(all_image_ids):
        missing_ids = pred_image_ids - set(all_image_ids)
        raise ValueError(
            f"Predictions contain image IDs not found in ground truth: {missing_ids}"
        )

    bar = all_image_ids

    list_iou = []
    list_ciou = []
    list_siou_sigma = []
    list_siou_2sigma = []
    list_siou_3sigma = []
    list_siou = []
    list_n_ratio = []
    list_boundary_iou = []
    pss = []
    sps = []
    sps_sigma = []
    sps_2sigma = []
    sps_3sigma = []
    results = []

    for image_id in bar:
        img_info = coco_gt.loadImgs(image_id)[0]
        image_name = os.path.basename(img_info["file_name"])

        # ---------------- GT mask & vertex count ----------------
        ann_ids_gt = coco_gt.getAnnIds(imgIds=img_info["id"])
        if not ann_ids_gt:
            # Skip images without GT polygons
            continue

        anns_gt = coco_gt.loadAnns(ann_ids_gt)

        mask_gt = None
        N_gt = 0
        for idx, ann in enumerate(anns_gt):
            m, n_v = _decode_coco_mask(
                ann, img_info["height"], img_info["width"]
            )
            if idx == 0:
                mask_gt = m
            else:
                mask_gt = mask_gt + m
            N_gt += n_v

        mask_gt = mask_gt != 0

        # ---------------- Pred mask & vertex count ----------------
        if image_id in pred_image_ids:
            ann_ids_dt = coco_dt.getAnnIds(imgIds=img_info["id"])
            anns_dt = coco_dt.loadAnns(ann_ids_dt)

            mask_pred = None
            N_pred = 0
            for idx, ann in enumerate(anns_dt):
                m, n_v = _decode_coco_mask(
                    ann, img_info["height"], img_info["width"]
                )
                if idx == 0:
                    mask_pred = m
                else:
                    mask_pred = mask_pred + m
                N_pred += n_v

            mask_pred = mask_pred != 0
        else:
            mask_pred = np.zeros(
                (img_info["height"], img_info["width"]), dtype=np.uint8
            )
            N_pred = 0

        # ---------------- IoU & Boundary IoU ----------------
        iou = calc_iou(mask_pred, mask_gt)

        boundary_pred = mask_to_boundary(mask_pred.astype(np.uint8), dilation_ratio=0.02)
        boundary_gt = mask_to_boundary(mask_gt.astype(np.uint8), dilation_ratio=0.02)
        boundary_iou = calc_iou(boundary_pred, boundary_gt)

        # ---------------- C-IoU (vertex efficiency) ----------------
        ps = 1.0 - np.abs(N_pred - N_gt) / (N_pred + N_gt + 1e-9)

        # ---------------- S-IoU (polygon simplicity) ----------------
        N0_sigma, N0_2sigma, N0_3sigma = 50, 90, 500

        sp_sigma = (1 + np.exp(0.1 * (3 - N0_sigma))) / (
            1 + np.exp(0.1 * (N_pred - N0_sigma))
        )
        sp_2sigma = (1 + np.exp(0.1 * (3 - N0_2sigma))) / (
            1 + np.exp(0.1 * (N_pred - N0_2sigma))
        )
        sp_3sigma = (1 + np.exp(0.1 * (3 - N0_3sigma))) / (
            1 + np.exp(0.1 * (N_pred - N0_3sigma))
        )

        siou_sigma = iou * sp_sigma
        siou_2sigma = iou * sp_2sigma
        siou_3sigma = iou * sp_3sigma
        siou = (siou_sigma + siou_2sigma + siou_3sigma) / 3.0

        # ---------------- N-ratio ----------------
        n_ratio = compute_n_ratio(N_pred, N_gt)

        # ---------------- Accumulate ----------------
        list_iou.append(iou)
        list_ciou.append(iou * ps)
        list_siou_sigma.append(siou_sigma)
        list_siou_2sigma.append(siou_2sigma)
        list_siou_3sigma.append(siou_3sigma)
        list_siou.append(siou)
        list_boundary_iou.append(boundary_iou)
        pss.append(ps)
        sps.append((sp_sigma + sp_2sigma + sp_3sigma) / 3.0)
        sps_sigma.append(sp_sigma)
        sps_2sigma.append(sp_2sigma)
        sps_3sigma.append(sp_3sigma)
        list_n_ratio.append(n_ratio)

        results.append(
            {
                "image_name": image_name,
                "n_ratio": round(float(n_ratio), 2) if not np.isnan(n_ratio) else None,
                "iou": round(float(iou), 2),
                "c_iou": round(float(iou * ps), 2),
                "boundary_iou": round(float(boundary_iou), 2),
                "s_iou": round(float(siou), 2),
            }
        )

    # ---------------- Final summary ----------------
    print("Done!")
    print("Mean IoU: ", np.mean(list_iou))
    print("Mean Boundary IoU: ", np.mean(list_boundary_iou))
    print("Mean C-IoU: ", np.mean(list_ciou))
    print("Mean S-IoU: ", np.mean(list_siou))
    print("Mean N-Ratio: ", np.nanmean(list_n_ratio))

    return np.mean(list_iou), np.mean(list_boundary_iou), np.mean(list_ciou), np.mean(list_siou), np.nanmean(list_n_ratio)

# ===================================================== #


@torch.no_grad()
def evaluate_iou(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    json_data = json.load(open('/irsa/ROAD_DATA/sat2graph_sn3_dataset/json_v2/test_v16.json', 'r'))
    id_to_filename = {img['id']: img['file_name'] for img in json_data['images']}  # 映射id和文件名对应

    torch.cuda.empty_cache()  # 视需要：每 N 次再清一次更合理
    # for samples, targets in metric_logger.log_every(data_loader, 10, header):

    # 全局累计交集和并集
    inter_1_total = 0
    union_1_total = 0

    inter_2_total = 0
    union_2_total = 0

    inter_3_total = 0
    union_3_total = 0

    inter_4_total = 0
    union_4_total = 0

    inter_5_total = 0
    union_5_total = 0

    def accumulate_iou_terms(pred, gt):
        """
        返回该图像的交集与并集，用于累计
        """
        pred_bin = pred > 0
        gt_bin = gt > 0

        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()

        return intersection, union

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        orig_target_sizes = orig_target_sizes.cpu().detach().numpy()
        for temp_j, result in enumerate(results):
            image_id = int(targets[temp_j]['image_id'].cpu().numpy())
            file_name = id_to_filename[image_id]
            gt = cv2.imread(os.path.join('/irsa/ROAD_DATA/sat2graph_sn3_dataset/p_mask_width3', file_name.replace('__rgb.png', '__gt.png')), 0)
            gt = ((gt > 0)*255).astype(np.uint8)
            pred_1 = np.zeros_like(gt)
            pred_2 = np.zeros_like(gt)
            pred_3 = np.zeros_like(gt)
            pred_4 = np.zeros_like(gt)
            pred_5 = np.zeros_like(gt)
            logits_np   = outputs['pred_logits'][temp_j].sigmoid()[..., 1].cpu().detach().numpy()
            pts_all_np  = outputs['pred_instances_pts'][temp_j].cpu().detach().numpy()
            # 阈值 & 对应的预测图
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
            pred_maps  = [pred_1, pred_2, pred_3, pred_4, pred_5]

            for one_point_score, one_instance_pts in zip(logits_np, pts_all_np):

                # 最低阈值都过不了就直接跳过，连坐标都不用算
                if one_point_score <= thresholds[0]:
                    continue

                # 只算一次坐标
                pts_px = (one_instance_pts * orig_target_sizes[temp_j]).astype(np.int32)

                # cv2.polylines 只用 C 端循环，比 Python for 快得多
                # 需要形状 (N, 1, 2)
                pts_px_poly = pts_px.reshape(-1, 1, 2)

                # 分别画在满足阈值的 pred_* 上
                for thr, pred_map in zip(thresholds, pred_maps):
                    if one_point_score > thr:
                        cv2.polylines(pred_map, [pts_px_poly], isClosed=False, color=255, thickness=3)
            # big = np.vstack([
            #     np.hstack([gt, pred_1]),
            #     np.hstack([pred_3, pred_5])
            # ])

            # # 画红色分界线（宽2像素）
            # H, W = gt.shape
            # cv2.line(big, (0, H), (2*W, H), (255,255,255), 2)   # 横线
            # cv2.line(big, (W, 0), (W, 2*H), (255,255,255), 2)   # 竖线
            # cv2.imwrite(os.path.join('/irsa/build_code/Deformable-DETR/main_road/vis', file_name.replace('__rgb.png', '_iou_vis.png')), big)
            # ---- 累计交集与并集 ----
            inter, union = accumulate_iou_terms(pred_1, gt)
            inter_1_total += inter
            union_1_total += union

            inter, union = accumulate_iou_terms(pred_2, gt)
            inter_2_total += inter
            union_2_total += union

            inter, union = accumulate_iou_terms(pred_3, gt)
            inter_3_total += inter
            union_3_total += union

            inter, union = accumulate_iou_terms(pred_4, gt)
            inter_4_total += inter
            union_4_total += union

            inter, union = accumulate_iou_terms(pred_5, gt)
            inter_5_total += inter
            union_5_total += union
        del samples, targets, outputs
    iou_1 = inter_1_total / union_1_total if union_1_total > 0 else 0
    iou_2 = inter_2_total / union_2_total if union_2_total > 0 else 0
    iou_3 = inter_3_total / union_3_total if union_3_total > 0 else 0
    iou_4 = inter_4_total / union_4_total if union_4_total > 0 else 0
    iou_5 = inter_5_total / union_5_total if union_5_total > 0 else 0

    print("Final IoU @ 0.1 =", iou_1)
    print("Final IoU @ 0.2 =", iou_2)
    print("Final IoU @ 0.3 =", iou_3)
    print("Final IoU @ 0.4 =", iou_4)
    print("Final IoU @ 0.5 =", iou_5)
    torch.cuda.empty_cache()  # 视需要：每 N 次再清一次更合理


def thr_eval(skel_dir, result_dict):
    # 像素级指标2 5 10个像素阈值下的准确率 召回率 F1分数
    def tuple2list(t):
            return [[t[0][x],t[1][x]] for x in range(len(t[0]))]
            
    gt_mask_dir = '/irsa/ROAD_DATA/sat2graph_sn3_dataset/p_mask_width1'
    pre_acc = np.zeros((4))
    pre_recall = np.zeros((4))
    pre_f1 = np.zeros((4))
    counter = 0
    img_list = [x.replace('__gt.png', '') for x in os.listdir(gt_mask_dir)]
    for i,img in enumerate(img_list):
        gt_image = np.array(Image.open(os.path.join(gt_mask_dir,img + '__gt.png')))
        if not os.path.exists(os.path.join(skel_dir,img + '__pred.png')):
            pre_image = np.zeros_like(gt_image)
        else:
            pre_image = np.array(Image.open(os.path.join(skel_dir,img + '__pred.png')))
        if len(np.where(pre_image!=0)[0])==0:
            continue
        gt_points = tuple2list(np.where(gt_image!=0))
        pre_points = tuple2list(np.where(pre_image!=0))
        gt_tree = cKDTree(gt_points)

        for ii,thr in enumerate([3,7,10]):  # [2,5,10]
            if len(pre_points):
                # recall
                pre_tree = cKDTree(pre_points)
                pre_dds,_ = pre_tree.query(gt_points, k=1)
                recall = len([x for x in pre_dds if x<thr])/len(pre_dds)
                pre_recall[ii] = (pre_recall[ii] * counter + recall) / (counter+1)
                # accuracy
                gt_acc_dds,_ = gt_tree.query(pre_points, k=1)
                acc = len([x for x in gt_acc_dds if x <thr])/len(gt_acc_dds)
                pre_acc[ii] = (pre_acc[ii] * counter + acc) / (counter+1)
                # f1 score
                if recall*acc:
                    f1 = 2*acc*recall/(acc+recall)
                else:
                    f1 = 0
                pre_f1[ii] = (pre_f1[ii] * counter + f1)/(counter+1)
        counter+=1
    result_dict.update({'threshold': os.path.basename(skel_dir), f'{os.path.basename(skel_dir)}_pre_acc_3':pre_acc.tolist()[0], f'{os.path.basename(skel_dir)}_pre_recall_3':pre_recall.tolist()[0], f'{os.path.basename(skel_dir)}_r_f1_3':pre_f1.tolist()[0], f'{os.path.basename(skel_dir)}_pre_acc_7':pre_acc.tolist()[1], f'{os.path.basename(skel_dir)}_pre_recall_7':pre_recall.tolist()[1], f'{os.path.basename(skel_dir)}_r_f1_7':pre_f1.tolist()[1], f'{os.path.basename(skel_dir)}_pre_acc_10':pre_acc.tolist()[2], f'{os.path.basename(skel_dir)}_pre_recall_10':pre_recall.tolist()[2], f'{os.path.basename(skel_dir)}_r_f1_10':pre_f1.tolist()[2]})
    return result_dict
    # print({'threshold': os.path.basename(skel_dir),'pre_acc':pre_acc.tolist(), 'pre_recall':pre_recall.tolist(), 'r_f1':pre_f1.tolist()})

def entropy_conn(skel_dir, result_dict):
    # 评估连接性指标 ECM 和 Naive
    gt_mask_dir = '/irsa/ROAD_DATA/sat2graph_sn3_dataset/p_mask_width1'
    img_list = [x.replace('__gt.png', '') for x in os.listdir(gt_mask_dir)]
    ECM = 0
    naive = 0
    for i,img in enumerate(img_list):
        gt_image = np.array(Image.open(os.path.join(gt_mask_dir,img + '__gt.png')))
        if not os.path.exists(os.path.join(skel_dir,img + '__pred.png')):  # 没有预测图则全0 也要评估
            pre_image = np.zeros_like(gt_image)
        else:
            pre_image = np.array(Image.open(os.path.join(skel_dir,img + '__pred.png')))
        # find instances of the gt map
        gt_instance_map = measure.label(gt_image / 255,background=0)
        gt_instance_indexes = np.unique(gt_instance_map)[1:]
        # record length of all predicted instance assigned to a gt instance
        gt_assigned_lengths = [[] for x in range(len(gt_instance_indexes))]
        # record gt-instance length and vertices of this instance
        gt_instance_length = []
        gt_instance_points = []
        # record gt-instance pixels covered by projected predicted instances (measure completion)
        gt_covered = []
        # each gt_index labels is an gt instance
        for index in gt_instance_indexes:
            instance_map = (gt_instance_map == index)
            instance_points = np.where(instance_map==1)
            instance_points = [[instance_points[0][i],instance_points[1][i]] for i in range(len(instance_points[0]))]
            gt_instance_length.append(len(instance_points))
            gt_covered.append(np.zeros((len(instance_points))))
            gt_instance_points.append(instance_points)
        # find instances of the predicted graph map
        pre_instance_map = measure.label(pre_image / 255,background=0)
        pre_instance_indexes = np.unique(pre_instance_map)[1:]
        # all gt pixel points
        gt_points = np.where(gt_image!=0)
        gt_points = [[gt_points[0][i],gt_points[1][i]] for i in range(len(gt_points[0]))]
        tree = cKDTree(gt_points)
        # each pre_index is an predicted instance
        for index in pre_instance_indexes: 
            votes = []
            instance_map = (pre_instance_map == index)
            instance_points = np.where(instance_map==1)
            instance_points = [[instance_points[0][i],instance_points[1][i]] for i in range(len(instance_points[0]))]
            if instance_points:
                # Each predicted point of the current pre-instance finds its closest gt point and votes
                # to the gt-instance that the closest gt point belongs to.
                _, iis = tree.query(instance_points,k=[1])
                closest_gt_points = [[gt_points[x[0]][0],gt_points[x[0]][1]] for x in iis]
                votes = [gt_instance_map[x[0],x[1]] for x in closest_gt_points]
            # count the voting results
            votes_summary = np.zeros((len(gt_instance_indexes)))
            for j in range(len(gt_instance_indexes)):
                # the number of votes made to gt-instance j+1
                votes_summary[j] = votes.count(j+1) 
            # find the gt-instance winning the most vote and assign the current pre-instance to it
            if np.max(votes_summary):
                vote_result = np.where(votes_summary==np.max(votes_summary))[0][0]
                # the length of the pre-instance assigned to corresponding gt-instance 
                gt_assigned_lengths[vote_result].append(len(instance_points))
                # calculate projection of the predicted instance to corresponding gt-instance
                instance_tree = cKDTree(gt_instance_points[vote_result])
                _, iis = instance_tree.query(instance_points,k=[1])
                gt_covered[vote_result][np.min(iis):np.max(iis)+1] = 1
        # calculate ECM
        entropy_conn = 0
        naive_conn = 0
        # iterate all gt-instances, calculate connectivity of each of them 
        for j,lengths in enumerate(gt_assigned_lengths):
            # lengths are the length of assigned pre-instances to the current gt-instance
            if len(lengths):
                lengths = np.array(lengths)
                # contribution of each assigned pre-instance
                probs = (lengths / np.sum(lengths)).tolist()
                C_j = 0
                for p in probs:
                    C_j += -p*np.log2(p)
                entropy_conn += np.exp(-C_j) * np.sum(gt_covered[j]) / len(gt_points)
                naive_conn += 1 / len(lengths)
        if len(gt_assigned_lengths):
            naive_conn = naive_conn / len(gt_assigned_lengths)
        # weighted sum
        ECM = (ECM * i + entropy_conn)/(i+1)
        naive = (naive * i + naive_conn)/(i+1)
    output_json = {f'{os.path.basename(skel_dir)}_ECM':np.array(ECM).tolist(),f'{os.path.basename(skel_dir)}_naive':naive}
    result_dict.update(output_json)
    return result_dict
        
@torch.no_grad()
def evaluate_pixeleval(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, epoch):
    model.eval()
    criterion.eval()
    
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    

    json_data = json.load(open('/irsa/ROAD_DATA/sat2graph_sn3_dataset/json_v2/test_v17.json', 'r'))
    id_to_filename = {img['id']: img['file_name'] for img in json_data['images']}  # 映射id和文件名对应

    torch.cuda.empty_cache()  # 视需要：每 N 次再清一次更合理

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6]
    root_path = os.path.join(output_dir, str(epoch), 'pixeleval')
    for t in thresholds: os.makedirs(f"{root_path}/{t}", exist_ok=True)

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        orig_target_sizes = orig_target_sizes.cpu().detach().numpy()
        for temp_j, result in enumerate(results):
            image_id = int(targets[temp_j]['image_id'].cpu().numpy())
            file_name = id_to_filename[image_id]
            pred_1 = np.zeros((400, 400))
            pred_2 = np.zeros_like(pred_1)
            pred_3 = np.zeros_like(pred_1)
            pred_4 = np.zeros_like(pred_1)
            pred_5 = np.zeros_like(pred_1)
            pred_55 = np.zeros_like(pred_1)
            pred_6 = np.zeros_like(pred_1)
            logits_np   = outputs['pred_logits'][temp_j].sigmoid()[..., 1].cpu().detach().numpy()
            pts_all_np  = outputs['pred_instances_pts'][temp_j].cpu().detach().numpy()
            # 阈值 & 对应的预测图
            pred_maps  = [pred_1, pred_2, pred_3, pred_4, pred_5, pred_55, pred_6]

            for one_point_score, one_instance_pts in zip(logits_np, pts_all_np):

                # 最低阈值都过不了就直接跳过，连坐标都不用算
                if one_point_score <= thresholds[0]:
                    continue

                # 只算一次坐标
                pts_px = (one_instance_pts * orig_target_sizes[temp_j]).astype(np.int32)

                # cv2.polylines 只用 C 端循环，比 Python for 快得多
                # 需要形状 (N, 1, 2)
                pts_px_poly = pts_px.reshape(-1, 1, 2)

                # 分别画在满足阈值的 pred_* 上
                for thr, pred_map in zip(thresholds, pred_maps):
                    if one_point_score > thr:
                        cv2.polylines(pred_map, [pts_px_poly], isClosed=False, color=255, thickness=1)
                    cv2.imwrite(os.path.join(root_path, str(thr), file_name.replace('__rgb', '__pred')), pred_map)
        del samples, targets, outputs
    torch.cuda.empty_cache()  # 视需要：每 N 次再清一次更合理
    if is_dist:
        dist.barrier()

    if rank == 0:  # 必须要限制 不然可能会评估多次
        for t in thresholds:
            result_dict = {}
            p = os.path.join(root_path, f"{t}")
            result_dict = thr_eval(p, result_dict)
            result_dict = entropy_conn(p, result_dict)
            print(result_dict)
                

@torch.no_grad()
def evaluate_apls(args, model, criterion, postprocessors, data_loader, base_ds, device, output_dir, epoch):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    json_data = json.load(open(args.coco_path + '/json/roadboundary_test_v5.json', 'r'))
    id_to_filename = {img['id']: img['file_name'] for img in json_data['images']}  # 映射id和文件名对应

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    graph_save_path = os.path.join(args.output_dir, str(epoch) + '_graphs')
    os.makedirs(graph_save_path, exist_ok=True) 


    torch.cuda.empty_cache()  # 视需要：每 N 次再清一次更合理
    # for samples, targets in metric_logger.log_every(data_loader, 10, header):
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # =============== 注释掉计算损失的部分 ================= #
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        # ================================================== #

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        # res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        for temp_j, result in enumerate(results):
            image_id = int(targets[temp_j]['image_id'].cpu().numpy())
            file_name = id_to_filename[image_id]
            # img = cv2.imread(os.path.join(args.coco_path, file_name))
            all_graph = {}
            for one_point_score, one_bbox, one_instance_pts in zip(outputs['pred_logits'][temp_j].sigmoid()[...,1].cpu().detach().numpy(), \
                                                                   outputs['pred_boxes'][temp_j].cpu().detach().numpy(), \
                                                                    outputs['pred_instances_pts'][temp_j].cpu().detach().numpy()):
                if one_point_score < 0.3:
                    continue
                # cv2.circle(img, (int(one_point_coord[0]), int(one_point_coord[1])), radius=3, color=(0, 255, 0), thickness=-1)
                # color = random.choice(color_palette)
                # for i in range(0, one_instance_pts.shape[0] -2):
                #     # cv2.circle(img, (int(one_instance_pts[i][0]*1000), int(one_instance_pts[i][1]*1000)), radius=3, color=color, thickness=-1)
                #     cv2.line(img, (int(one_instance_pts[i][0]*1000), int(one_instance_pts[i][1]*1000)), (int(one_instance_pts[i+1][0]*1000), int(one_instance_pts[i+1][1]*1000)), color=color, thickness=2)
                pts_px = (one_instance_pts * 1000.0).astype(np.float32)
                graph = build_undirected_graph(pts_px, wrap_threshold=5.0)
                all_graph = merge_graph_dicts(all_graph, graph)
                # one_bbox = box_cxcywh_to_xyxy(one_bbox)
                # cv2.rectangle(img, (int((one_bbox[0] - one_bbox[2] /2)*1000), int((one_bbox[1] - one_bbox[3] /2)*1000)), (int((one_bbox[0] + one_bbox[2] /2)*1000), int((one_bbox[1] + one_bbox[3] /2)*1000)), color=color, thickness=2)
            # cv2.imwrite(os.path.join(args.show_path, file_name.replace('cropped_tiff/', '').replace('.tiff', '.png')), img)
            out_p = os.path.join(
                graph_save_path,
                file_name.replace('cropped_tiff/', '').replace('.tiff', '_graphs.p')
            )
            with open(out_p, 'wb') as f:
                pickle.dump(all_graph, f, protocol=pickle.HIGHEST_PROTOCOL)

        del samples, targets, outputs

    now_dir = os.getcwd()
    os.chdir("/irsa/Go_setup/road_metric/topoboundary_metrics/apls")

    graph_dir = graph_save_path

    t1 = time.time()
    subprocess.run(['python', 'convert.py', graph_dir])
    t2 = time.time()
    # print("convert time:", t2 - t1)

    subprocess.run(["go", "run", "main_multthread_onlineeval.go", "-dir", graph_dir])
    t3 = time.time()
    # print("go eval time:", t3 - t2)

    subprocess.run(['python', 'read_txt_online.py', graph_dir])
    t4 = time.time()
    # print("read txt time:", t4 - t3)

    os.chdir(now_dir)
    torch.cuda.empty_cache()  # 视需要：每 N 次再清一次更合理

@torch.no_grad()
def evaluate_aplsv2(args, model, criterion, postprocessors, data_loader, base_ds, device, output_dir, epoch):
    model.eval()
    criterion.eval()

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    json_data = json.load(open(args.coco_path + '/json/roadboundary_test_v5.json', 'r'))
    id_to_filename = {img['id']: img['file_name'] for img in json_data['images']}  # 映射id和文件名对应

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    graph_save_path = os.path.join(args.output_dir, str(epoch) + '_graphs')
    os.makedirs(graph_save_path, exist_ok=True) 


    torch.cuda.empty_cache()  # 视需要：每 N 次再清一次更合理
    # for samples, targets in metric_logger.log_every(data_loader, 10, header):
    with torch.inference_mode(): 
        for samples, targets in data_loader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            # =============== 注释掉计算损失的部分 ================= #
            # loss_dict = criterion(outputs, targets)
            # weight_dict = criterion.weight_dict

            # # reduce losses over all GPUs for logging purposes
            # loss_dict_reduced = utils.reduce_dict(loss_dict)
            # loss_dict_reduced_scaled = {k: v * weight_dict[k]
            #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
            # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
            #                               for k, v in loss_dict_reduced.items()}
            # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
            #                      **loss_dict_reduced_scaled,
            #                      **loss_dict_reduced_unscaled)
            # metric_logger.update(class_error=loss_dict_reduced['class_error'])
            # ================================================== #

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            if 'segm' in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            # res = {target['image_id'].item(): output for target, output in zip(targets, results)}

            for temp_j, result in enumerate(results):
                image_id = int(targets[temp_j]['image_id'].cpu().numpy())
                file_name = id_to_filename[image_id]
                # img = cv2.imread(os.path.join(args.coco_path, file_name))
                all_graph = {}
                for one_point_score, one_bbox, one_instance_pts in zip(outputs['pred_logits'][temp_j].sigmoid().cpu().detach().numpy(), \
                                                                    outputs['pred_boxes'][temp_j].cpu().detach().numpy(), \
                                                                        outputs['pred_instances_pts'][temp_j].cpu().detach().numpy()):
                    index, one_score = np.argmax(one_point_score), one_point_score[np.argmax(one_point_score)]
                    if one_score < 0.3:
                        continue
                    # cv2.circle(img, (int(one_point_coord[0]), int(one_point_coord[1])), radius=3, color=(0, 255, 0), thickness=-1)
                    # color = random.choice(color_palette)
                    # for i in range(0, one_instance_pts.shape[0] -2):
                    #     # cv2.circle(img, (int(one_instance_pts[i][0]*1000), int(one_instance_pts[i][1]*1000)), radius=3, color=color, thickness=-1)
                    #     cv2.line(img, (int(one_instance_pts[i][0]*1000), int(one_instance_pts[i][1]*1000)), (int(one_instance_pts[i+1][0]*1000), int(one_instance_pts[i+1][1]*1000)), color=color, thickness=2)
                    pts_px = one_instance_pts * 1000.0
                    is_ring = (index == 2)
                    graph = build_undirected_graph_v2(pts_px, is_ring)
                    all_graph = merge_graph_dicts(all_graph, graph)
                    # one_bbox = box_cxcywh_to_xyxy(one_bbox)
                    # cv2.rectangle(img, (int((one_bbox[0] - one_bbox[2] /2)*1000), int((one_bbox[1] - one_bbox[3] /2)*1000)), (int((one_bbox[0] + one_bbox[2] /2)*1000), int((one_bbox[1] + one_bbox[3] /2)*1000)), color=color, thickness=2)
                # cv2.imwrite(os.path.join(args.show_path, file_name.replace('cropped_tiff/', '').replace('.tiff', '.png')), img)
                out_p = os.path.join(
                    graph_save_path,
                    file_name.replace('cropped_tiff/', '').replace('.tiff', '_graphs.p')
                )
                with open(out_p, 'wb') as f:
                    pickle.dump(all_graph, f, protocol=pickle.HIGHEST_PROTOCOL)

            del samples, targets, outputs

    torch.cuda.empty_cache()  # 视需要：每 N 次再清一次更合理
    if is_dist:
        dist.barrier()

    if rank == 0:  # 必须要限制 不然可能会评估多次
        now_dir = os.getcwd()
        os.chdir("/irsa/Go_setup/road_metric/topoboundary_metrics/apls")

        graph_dir = graph_save_path

        t1 = time.time()
        subprocess.run(['python', 'convert.py', graph_dir])
        t2 = time.time()
        # print("convert time:", t2 - t1)

        subprocess.run(["go", "run", "main_multthread_onlineeval.go", "-dir", graph_dir])
        t3 = time.time()
        # print("go eval time:", t3 - t2)

        subprocess.run(['python', 'read_txt_online.py', graph_dir])
        t4 = time.time()
        # print("read txt time:", t4 - t3)

        os.chdir(now_dir)
        torch.cuda.empty_cache()  # 视需要：每 N 次再清一次更合理
