#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from typing import Optional, Tuple
import torch
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from parcnet.cvnets.models.detection.base_detection import DetectionPredTuple

from utils.tensor_utils import tensor_to_python_float

from .topk_accuracy import top_k_accuracy
from .intersection_over_union import compute_miou_batch


def metric_monitor(pred_label: Tensor or Tuple[Tensor], target_label: Tensor, loss: Tensor or float, metric_names: list,
                   use_distributed: Optional[bool] = False):
    metric_vals = dict()
    if "loss" in metric_names:
        loss = tensor_to_python_float(loss, is_distributed=use_distributed)
        metric_vals['loss'] = loss

    if "top1" in metric_names:
        top_1_acc, top_5_acc = top_k_accuracy(pred_label, target_label, top_k=(1, 5))
        top_1_acc = tensor_to_python_float(top_1_acc, is_distributed=use_distributed)
        metric_vals['top1'] = top_1_acc
        if "top5" in metric_names:
            top_5_acc = tensor_to_python_float(top_5_acc, is_distributed=use_distributed)
            metric_vals['top5'] = top_5_acc

    if "iou" in metric_names:
        inter, union = compute_miou_batch(prediction=pred_label, target=target_label)

        inter = tensor_to_python_float(inter, is_distributed=use_distributed)
        union = tensor_to_python_float(union, is_distributed=use_distributed)
        metric_vals['iou'] = {
            'inter': inter,
            'union': union
        }
    
    # if "map" in metric_names:
        
    #     confidence, predicted_locations, _ = pred_label
    #     metric = MeanAveragePrecision()
    #     preds = []
    #     targets = []
    #     batch_size = target_label['box_labels'].shape[0]
        
    #     num_coordinates = predicted_locations.shape[-1]
        
    #     for i in range(batch_size):
    #         num_classes = confidence.shape[-1]

    #         gt_labels = target_label["box_labels"][i]
    #         gt_locations = target_label["box_coordinates"][i]

    #         pos_mask = gt_labels > 0

    #         gt_locations = gt_locations[pos_mask, :].view(-1, num_coordinates)
            
    #         targets.append({
    #             'boxes': gt_locations,
    #             'scores': torch.ones(gt_labels.shape[0]),
    #             'labels': gt_labels,
    #         })

    #         scores = torch.nn.functional.softmax(confidence[i], dim=-1)
    #         scores = scores[0]

    #         predicted_locations = predicted_locations[i][pos_mask, :].view(-1, num_coordinates)

    #         object_labels: list[list] = []
    #         object_scores: list[Tensor] = []
    #         for class_index in range(1, num_classes):
    #             score = scores[class_index]
    #             keep_idxs = score > 0.3
    #             score = score[keep_idxs]

    #             object_scores.append(score)
    #             object_labels.append(torch.full_like(score, fill_value=class_index, dtype=torch.int8,))
            
    #         preds.append({
    #             'boxes': predicted_locations[i],
    #             'scores':torch.cat(object_scores, dim=0),
    #             'labels':torch.cat(object_labels, dim=0)
    #         })

    #     print(len(targets), targets[0])
    #     print(len(preds), preds[0])
    #     metric.update(preds, targets)
    #     result_map: Tensor = metric.compute()['map']
    #     print(result_map)

    #     metric_vals['map'] = float(result_map.cpu().item())

    return metric_vals
