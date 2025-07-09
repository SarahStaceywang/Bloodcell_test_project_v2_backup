### structure
## BCCD_Dataset
BCCD数据集中的图片已转为COCO格式（json文件）储存在YOLOX/datasets/BCCD
## scripts 
这里中包含一个修复json文件中标注格式的命令，之前是字符串，改成了字符，应该没问题了。


## voc2coco
已完成，将jpg格式的图片转为json格式文件。
命令如下，之后有图片格式转换需求可以参考：
```bash
$ python voc2coco.py --ann_dir ../BCCD_Dataset/BCCD/Annotations --ann_ids ../BCCD_Dataset/BCCD/ImageSets/Main/train.txt --labels ../BCCD_Dataset/labels.txt --output ../YOLOX/datasets/BCCD/annotations/instances_train2017.json --ext xml --img_dir ../BCCD_Dataset/JPEGImages

$ python voc2coco.py --ann_dir ../BCCD_Dataset/BCCD/Annotations --ann_ids ../BCCD_Dataset/BCCD/ImageSets/Main/val.txt --labels ../BCCD_Dataset/labels.txt --output ../YOLOX/datasets/BCCD/annotations/instances_val2017.json --ext xml

$ mkdir -p ../YOLOX/datasets/BCCD/images/train2017
$ mkdir -p ../YOLOX/datasets/BCCD/images/val2017

# 读取ID列表并复制文件
$ xargs -a ../BCCD_Dataset/ImageSets/Main/train.txt -I{} cp ../BCCD_Dataset/JPEGImages/{}.jpg ../YOLOX/datasets/BCCD/images/train2017/
$ xargs -a ../BCCD_Dataset/ImageSets/Main/val.txt -I{} cp ../BCCD_Dataset/JPEGImages/{}.jpg ../YOLOX/datasets/BCCD/images/val2017/

while read p; do 
    cp JPEGImages/$p.jpg ../YOLOX/datasets/BCCD/images/train2017/;
done <ImageSets/Main/train.txt

 Get-Content BCCD_Dataset/BCCD/ImageSets/Main/train.txt | ForEach-Object {
    Copy-Item "BCCD_Dataset/BCCD/JPEGImages/$_.jpg" "../bloodcelld_test_project/YOLOX/datasets/BCCD/images/train2017/"
}

Get-Content BCCD_Dataset/BCCD/ImageSets/Main/val.txt | ForEach-Object {
    Copy-Item "BCCD_Dataset/BCCD/JPEGImages/$_.jpg" "../bloodcelld_test_project/YOLOX/datasets/BCCD/images/val2017/"
}
```
应该是这样。

## YOLOX
YOLOX/
├── datasets/
│   └── BCCD/           # COCO格式的BCCD数据集
│       ├── annotations/
│       │   ├── instances_train2017_fixed.json  # 用scriptsfixed过的训练集。要用这个和下面那个验证集。
│       │   └── instances_val2017_fixed.json  # fixed过的验证集。用这个。
│           ├── instances_train2017.json # 这俩是没改过的。
│           └── instances_val2017.json
│       ├── images/ 
│       ├── train2017/
│       └── val2017/
├── exps/
│   └── example/
│        └── custom/
│           ├── yolox_bccd_nano_test.py        # 这个应该是可以运行的，路径是对的，参数要改。
│           ├── yolox_bccd_nano.py             # 这个没改好，和上面功能一致
│           ├── nano.py                    # 忘了从哪来的，好像是YOLOX/README.md里下载的，或者是官网下载的。
│           ├── yolox_s.py               # 和nano.py一个来源应该是
├── Real_README.md  <-- You are here

这个YOLOX一共提供了六种不同体量的模型，我这里用的是最轻量化的nano
其他的有关YOLOX本身的操作、资料都在YOLOX/README.md文件里

## steps
### 1.配置环境
```bash
$ git clone git@github.com:SarahStaceywang/Bloodcell_test_project_v2.git
cd YOLOX
pip3 install -v -e .
```
```bash
$ cd bloodcelld_test_project
conda create -n bloodcelld_test_project python=3.8 -y
conda activate bloodcelld_test_project

pip install -U pip
pip install -r requirements.txt  
pip install opencv-python pycocotools matplotlib tqdm loguru tensorboard
pip install -U albumentations cython onnx onnxruntime scikit-image
```

### 2.开始运行
# 若从头开始训练
```bash
$ cd YOLOX
python tools/train.py -f exps/example/custom/yolox_bccd_nano_test.py -d 1 -b 8 --fp16 -o
```
-f：指定配置文件路径。
-d 1：使用一个 GPU。
-b 4：批大小为 4。
--fp16：使用混合精度训练（如果您的 GPU 支持）。
-o：启用优化器。 
训练过程中，模型权重将保存在YOLOX\YOLOX_outputs\yolox_bccd_nano_test

# 若使用预训练权重
# 1.使用yolox_nano模型
测试一下地址对不对
```bash
$ python tools/train.py -f exps/example/yolox_base_bccd_nano.py -d 1 -b 8 --fp16 -o -c exps/example/custom/yolox_nano.pth --resume
```
开始运行
```bash
$ python tools/train.py -f exps/example/yolox_base_bccd_nano.py -d 1 -b 8 --fp16 -o -c exps/example/custom/yolox_nano.pth --cache
```

# 2.使用yolox_s模型
```bash
$ python tools/train.py -f exps/example/yolox_base_bccd_s.py -d 2 -b 16 --fp16 -o -c exps/example/custom/yolox_s.pth --cache
```

### 3.模型评估
```bash
$ python tools/eval.py -n yolox_base_bccd_nano -c YOLOX_outputs/yolox_base_bccd_nano/best_ckpt.pth -b 8 -d 1 --conf 0.001 --fp16

#数字结果
python tools/eval.py -f exps/example/yolox_base_bccd_nano.py -c YOLOX_outputs/yolox_base_bccd_nano/best_ckpt.pth -b 8 -d 1 --conf 0.001 --fp16

#混淆矩阵
python eval_try.py -f exps/example/yolox_base_bccd_nano.py -c YOLOX_outputs/yolox_base_bccd_nano/best_ckpt.pth -b 8 -d 1 --conf 0.001 --fp16

python eval_try.py -f exps/example/yolox_base_bccd_s.py -c YOLOX_outputs/yolox_base_bccd_s/best_ckpt.pth -b 8 -d 1 --conf 0.001 --fp16
```
2025/7/1 修改了eval_try.py,原代码在/YOLOX/eval_try_copy.py中
remark.
画混淆矩阵时需要把coco_evaluator.py中的
def evaluate(
        self, model, distributed=False, half=False, trt_file=None,
        decoder=None, test_size=None, return_outputs=False #改成True画混淆矩阵
    )

-n：模型名称。
-c：模型权重文件路径。
-b：批大小。
-d：使用的 GPU 数量。
--conf：置信度阈值。
--fp16：使用混合精度

### 4.可视化
```bash
$ python3 tools/visualize_assign.py -f exps/example/yolox_base_bccd_nano.py -c YOLOX_outputs/yolox_bccd_nano_test/best_ckpt.pth -d 1 -b 8 --max-batch 1
```