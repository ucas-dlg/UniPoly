#  对于dp简化后的建筑物进行ap精度评估
import numpy as np
import torch

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from .coco_eval import CocoEvaluator

from util.misc import all_gather

class BuildingEvaluator(CocoEvaluator):
    def prepare_for_coco_polygon(self, predictions):  # add by DLG
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]


            # masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            bboxes = prediction["boxes"].tolist()

            # rles = [
            #     mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
            #     for mask in masks
            # ]
            # for rle in rles:
            #     rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": [[x for x in polygon_building.tolist() if x != -1.0]],  # list of x,y
                        "score": scores[k],
                        "bbox": bboxes[k],
                    }
                    for k, polygon_building in enumerate(masks)
                ]
            )
        return coco_results