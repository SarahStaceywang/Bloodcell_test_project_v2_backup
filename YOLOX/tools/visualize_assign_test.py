import os
import sys
import time
import random
import warnings
import torch
import torch.backends.cudnn as cudnn
from loguru import logger

from yolox.exp import get_exp
from yolox.utils import configure_module, configure_omp, postprocess
from tools.train import make_parser
from yolox.data.datasets.coco_classes import COCO_CLASSES

import cv2
import numpy as np


def vis(img, boxes, scores, cls_ids, conf=0.3, class_names=None):
    img = np.ascontiguousarray(img)
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0, y0, x1, y1 = [int(i) for i in box]
        color = (0, 255, 0)
        text = f"{class_names[cls_id]}:{score:.1%}" if class_names else f"{cls_id}:{score:.1%}"
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img, text, (x0, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn("CUDNN deterministic mode will slow down training.")

    configure_omp()
    cudnn.benchmark = True

    # === 创建模型并载入权重 ===
    model = exp.get_model()
    model.eval()
    ckpt_file = args.ckpt or os.path.join(exp.output_dir, args.experiment_name, "best_ckpt.pth")
    logger.info(f"Loading checkpoint from {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda()

    # === 使用 exp 提供的 loader 加载验证集，禁用增强 ===
    val_loader = exp.get_eval_loader(
        batch_size=args.batch_size,
        is_distributed=False,
        testdev=False,
        no_aug=True
    )

    os.makedirs("vis_outputs", exist_ok=True)

    class_names = COCO_CLASSES
    max_vis = 10 #不太懂
    vis_count = 0

    for batch_idx, (imgs, targets, info_imgs, img_ids) in enumerate(val_loader):
        if vis_count >= max_vis:
            break

        imgs = imgs.cuda()
        with torch.no_grad():
            outputs = model(imgs)
            outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)

        for i in range(len(imgs)):
            if vis_count >= max_vis:
                break

            output = outputs[i]
            if output is None:
                continue

            output = output.cpu().numpy()
            boxes = output[:, 0:4]
            scores = output[:, 4] * output[:, 5]
            cls_ids = output[:, 6]

            img_info = info_imgs[i]
            # raw_h = img_info["height"]
            # raw_w = img_info["width"]
            raw_h = 480
            raw_w = 640
            input_h, input_w = imgs.shape[2:]  # 模型输入 896x896
            # 1. 获取缩放比例
            r = min(input_h / raw_h, input_w / raw_w)  # 缩放比例
            resized_h = int(raw_h * r)
            resized_w = int(raw_w * r)
            
            # 2. padding（letterbox）大小
            pad_h = (input_h - resized_h) / 2
            pad_w = (input_w - resized_w) / 2
            
            # 3. 将 box 还原到原图尺度（先减 padding 再除以缩放）
            boxes[:, 0::2] = (boxes[:, 0::2] ) / r  # x1, x2
            boxes[:, 1::2] = (boxes[:, 1::2] ) / r  # y1, y2
            
            # 4. clip 超出边界的框（防止负数或超过原图尺寸）
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, raw_w)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, raw_h)           
        
            # # 若存在 ratio，可以继续使用；否则根据输入尺寸和原图尺寸手动计算
            # input_h, input_w = imgs.shape[2:]  # 网络输入尺寸 (e.g., 640x640)
            # ratio_h = raw_h / input_h
            # ratio_w = raw_w / input_w
            # print("ratio_h:", ratio_h, "ratio_w:", ratio_w)

            # # 如果模型是等比缩放，ratio_h == ratio_w；否则需要分别缩放
            # boxes[:, 0::2] *= ratio_w
            # boxes[:, 1::2] *= ratio_h

            # 获取原始图路径
            img_path = val_loader.dataset.coco.loadImgs(int(img_ids[i]))[0]["file_name"]
            data_root = "/media/sata4/hzh/bccd/Bloodcell_test_project_v2/YOLOX/datasets/BCCD/images/val2017"
            full_img_path = os.path.join(data_root, img_path)

            if not os.path.exists(full_img_path):
                logger.warning(f"Image not found: {full_img_path}")
                continue
            ori_img = cv2.imread(full_img_path)
            if ori_img is None:
                logger.warning(f"Failed to read image: {full_img_path}")
                continue
            #full_img_path = img_path

            #ori_img = cv2.imread(full_img_path)

            result = vis(ori_img, boxes, scores, cls_ids, conf=exp.test_conf, class_names=class_names)

            save_path = f"vis_outputs/vis_{vis_count:03d}.jpg"
            cv2.imwrite(save_path, result)
            logger.info(f"Saved: {save_path}")
            vis_count += 1


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    main(exp, args)