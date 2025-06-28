#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO

import importlib.util
import cv2

from yolox.utils import postprocess

# ------------- 配置参数 -------------
EXP_FILE = "exps/example/yolox_base_bccd_nano.py"  # 自定义 exp 文件
CKPT_FILE = "YOLOX_outputs/yolox_base_bccd_nano/best_ckpt.pth"
IMG_DIR = "datasets/BCCD/val2017"             # 验证集图片目录
JSON_GT = "datasets/BCCD/annotations/fixed_instances_val2017_fixed_copy.json"  # GT JSON
CLASS_NAMES = ["RBC", "WBC", "Platelets"]
NUM_CLASSES = len(CLASS_NAMES)
SCORE_THRESH = 0.3
IOU_THRESH = 0.5
INPUT_SIZE = (640, 640)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------

# 加载自定义 Exp
spec = importlib.util.spec_from_file_location(
    "exp_module", os.path.abspath(EXP_FILE)
)
exp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp_module)
exp = exp_module.Exp()

# 加载模型和权重
model = exp.get_model().to(DEVICE)
ckpt = torch.load(CKPT_FILE, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

# 加载 COCO 验证集
coco_gt = COCO(JSON_GT)
img_ids = coco_gt.getImgIds()

# 初始化混淆矩阵
conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

# IoU 计算函数
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter + 1e-9)

# 开始评估
for img_id in tqdm(img_ids, desc="Evaluating"):
    img_info = coco_gt.loadImgs(img_id)[0]
    file_path = os.path.join(IMG_DIR, img_info["file_name"])
    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, INPUT_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).float().to(DEVICE) / 255.0

    with torch.no_grad():
        outputs = model(img_tensor)
        outputs = postprocess(outputs, NUM_CLASSES, SCORE_THRESH, IOU_THRESH)[0]

    # 真值框
    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    gt_anns = coco_gt.loadAnns(ann_ids)
    gt_boxes = [
        (ann["bbox"], ann["category_id"]) for ann in gt_anns if ann["category_id"] < NUM_CLASSES
    ]
    gt_boxes = [[x,y,x+w,y+h] for (x,y,w,h),cid in gt_boxes]
    gt_labels = [ann["category_id"] for ann in gt_anns if ann["category_id"] < NUM_CLASSES]

    if outputs is None:
        continue

    preds = outputs.cpu().numpy()
    pred_boxes = preds[:, :4]
    pred_labels = preds[:, 6].astype(int)

    matched = set()
    for pb, pl in zip(pred_boxes, pred_labels):
        best_iou = 0; best_idx = -1
        for i, gb in enumerate(gt_boxes):
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou; best_idx = i
        if best_iou >= IOU_THRESH and best_idx not in matched:
            conf_matrix[gt_labels[best_idx], pl] += 1
            matched.add(best_idx)

# 绘制混淆矩阵
import seaborn as sns
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.title("3×3 Confusion Matrix (IOU>=0.5)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()