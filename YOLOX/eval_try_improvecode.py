#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import (
    configure_module,
    configure_nccl,
    fuse_model,
    get_local_rank,
    get_model_info,
    setup_logger
)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
def bbox_iou(box1, box2):
    """
    IoU计算两个框的交并比
    box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def generate_confusion_matrix(output_data, dataset, save_path="confusion_matrix.png", iou_threshold=0.5):
    y_true = []
    y_pred = []

    coco = dataset.coco
    imgid_to_anns = coco.imgToAnns

    for img_id, pred in output_data.items():
        pred_boxes = pred["bboxes"]
        pred_classes = pred["categories"]
        gt_anns = imgid_to_anns[img_id]
        gt_boxes = [ann["bbox"] for ann in gt_anns]
        gt_classes = [ann["category_id"] for ann in gt_anns]

        # COCO 格式 [x, y, w, h] → [x1, y1, x2, y2]
        gt_boxes = [[x, y, x + w, y + h] for x, y, w, h in gt_boxes]

        matched_gt = set()
        matched_pred = set()

        for pred_idx, (pred_box, pred_class) in enumerate(zip(pred_boxes, pred_classes)):
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                if gt_idx in matched_gt:
                    continue
                iou = bbox_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx != -1:
                y_true.append(gt_classes[best_gt_idx])
                y_pred.append(pred_class)
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)

        # 漏检（GT 没有被任何预测框匹配）
        for gt_idx, gt_class in enumerate(gt_classes):
            if gt_idx not in matched_gt:
                y_true.append(gt_class)
                y_pred.append(-1)  # -1 表示漏检

        # 错检（预测框没有任何 GT 匹配）
        for pred_idx, pred_class in enumerate(pred_classes):
            if pred_idx not in matched_pred:
                y_true.append(-1)
                y_pred.append(pred_class)

    labels = sorted([cat["id"] for cat in coco.cats.values()])
    class_names = [coco.cats[l]["name"] for l in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels + [-1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names + ["missed"])

    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(xticks_rotation=45, ax=ax, cmap='Blues')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {save_path}")


def main(exp, args):
    file_name = os.path.join(exp.output_dir, "eval")
    os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed=False, filename="eval_log.txt", mode="a")

    model = exp.get_model()
    model.eval()

    if args.ckpt is None:
        ckpt_file = os.path.join(exp.output_dir, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.cuda()

    if args.fuse:
        model = fuse_model(model)

    evaluator = COCOEvaluator(
        dataloader=exp.get_eval_loader(args.batch_size, is_distributed=False),
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
    )

    output_data = defaultdict(lambda: {"bboxes": [], "categories": []})

    for (imgs, _, info_imgs, ids) in tqdm(evaluator.dataloader):
        imgs = imgs.cuda()
        with torch.no_grad():
            outputs = model(imgs)

        for output, img_id, info_img in zip(outputs, ids, info_imgs):
            if output is None:
                continue
            bboxes = output[:, :4].cpu().numpy()
            scores = output[:, 4] * output[:, 5]
            classes = output[:, 6].int().cpu().numpy()

            # 将 box 坐标从 [x1,y1,x2,y2] 转为图像尺寸比例
            for box, cls in zip(bboxes, classes):
                output_data[int(img_id)]["bboxes"].append(box.tolist())
                output_data[int(img_id)]["categories"].append(int(cls))

    generate_confusion_matrix(output_data, evaluator.dataloader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("YOLOX Evaluation with Confusion Matrix")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--fuse", action="store_true", default=False)
    parser.add_argument("--exp_file", default=None, type=str)
    parser.add_argument("--nms", type=float, default=0.65)
    parser.add_argument("--conf", type=float, default=0.01)

    args = parser.parse_args()
    exp = get_exp(args.exp_file, None)
    exp.test_conf = args.conf
    exp.nmsthre = args.nms

    launch(
        main,
        num_gpus=1,
        args=(exp, args),
    )