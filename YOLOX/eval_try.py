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
from collections import defaultdict

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
    # box 格式为 [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def generate_confusion_matrix(output_data, dataset, save_path="confusion_matrix/yolox_s/confusion_matrix_nano_0.3_test2(0.1).png", iou_threshold=0.3):
    y_true = []
    y_pred = []

    coco = dataset.coco
    imgid_to_anns = coco.imgToAnns

    # 获取类别id映射（如COCO类别id不是0,1,2,3...）
    labels = sorted([cat["id"] for cat in coco.cats.values()])
    id2idx = {cat_id: idx for idx, cat_id in enumerate(labels)}

    for img_id, pred in output_data.items():
        pred_boxes = pred["bboxes"]  # 预测框，格式为 [[x1, y1, x2, y2], ...]
        pred_classes = pred["categories"]  # 预测类别
        # 如果有scores字段，过滤低置信度框
        if "scores" in pred:
            pred_scores = pred["scores"]
            keep = [i for i, s in enumerate(pred_scores) if s > 0.3]
            pred_boxes = [pred_boxes[i] for i in keep]
            pred_classes = [pred_classes[i] for i in keep]

        gt_anns = imgid_to_anns[img_id]  # 真实标注
        gt_boxes = [ann["bbox"] for ann in gt_anns]  # 真实框，格式为 [x, y, w, h]
        gt_classes = [ann["category_id"] for ann in gt_anns]  # 真实类别

        # 将 COCO 格式的 bbox 转换为 [x1, y1, x2, y2]
        gt_boxes = [[x, y, x + w, y + h] for x, y, w, h in gt_boxes]
        # pred_boxes = [[x1, y1, x2, y2] for x1, y1, x2, y2 in pred_boxes]

         # 检查预测类别是否需要映射
        # 如果模型输出类别是0,1,2,3...，而COCO id不是，则需要映射
        if min(pred_classes, default=0) == 0 and min(labels) != 0:
            pred_classes = [labels[c] for c in pred_classes]
        
        # IoU 匹配
        matched_gt = set()
        matched_pred = set()
        
        if len(pred_boxes) > 0 and (pred_boxes[0][2] < pred_boxes[0][0] or pred_boxes[0][3] < pred_boxes[0][1]):
            pred_boxes = [[x, y, x + w, y + h] for x, y, w, h in pred_boxes]

        # 调试输出
        # print(f"img_id: {img_id}, pred_boxes: {pred_boxes}, pred_classes: {pred_classes}, gt_boxes: {gt_boxes}, gt_classes: {gt_classes}")


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

    # labels = sorted([cat["id"] for cat in coco.cats.values()])
    class_names = [coco.cats[l]["name"] for l in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels + [-1])  # 包含漏检类别
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names + ["missed"])

    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(xticks_rotation=45, ax=ax, cmap='Blues')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {save_path}")



def make_parser():

    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    logger.info("Model Structure:\n{}".format(str(model)))

    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True


    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint from {}".format(ckpt_file))
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc, weights_only=False)
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # start evaluate
    eval_results, output_data = evaluator.evaluate(
    model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, return_outputs=True
)
      # Generate confusion matrix after evaluation
    generate_confusion_matrix(output_data, evaluator.dataloader.dataset)
    
    # *_, summary = evaluator.evaluate(
    #     model, is_distributed, args.fp16, trt_file, decoder, exp.test_size
    # )
    # logger.info("\n" + summary)


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )