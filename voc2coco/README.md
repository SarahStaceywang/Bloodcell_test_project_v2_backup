# voc2coco

This is script for converting VOC format XMLs to COCO format json(ex. coco_eval.json).

### Why we need to convert VOC xmls to COCO format json ?

We can use COCO API, this is very useful(ex. calculating mAP).

## How to use

### 1. Make labels.txt

labels.txt if need for making dictionary for converting label to id.

**Sample labels.txt**

```txt
Label1
Label2
...
```

In order to get all labels from your `*.xml` files, you can use this command in shell:

```
grep -REoh '<name>.*</name>' /Path_to_Folder | sort | uniq
```

This will search for all name tags in `VOC.xml` files, then show unique ones. You can also go further and create `labels.txt` file. 

```
grep -ERoh '<name>(.*)</name>' /Path_to_folder | sort | uniq | sed 's/<name>//g' | sed 's/<\/name>//g' > labels.txt
```


### 2. Run script

##### 2.1 Usage 1(Use ids list)

```bash
$ python voc2coco.py --ann_dir /path/to/annotation/dir --ann_ids /path/to/annotations/ids/list.txt --labels /path/to/labels.txt --output /path/to/output.json <option> --ext xml
```

##### 2.2 Usage 2(Use annotation paths list)

**Sample paths.txt**

```txt
/path/to/annotation/file.xml
/path/to/annotation/file2.xml
...
```

```bash
$ python voc2coco.py \
    --ann_paths_list /path/to/annotation/paths.txt \
    --labels /path/to/labels.txt \
    --output /path/to/output.json \
    <option> --ext xml
```

### 3. Example of usage

In this case, you can convert [Shenggan/BCCD_Dataset: BCCD Dataset is a small-scale dataset for blood cells detection.](https://github.com/Shenggan/BCCD_Dataset) by this script.

```bash
$ python voc2coco.py --ann_dir sample/Annotations --ann_ids sample/dataset_ids/test.txt --labels sample/labels.txt --output sample/bccd_test_cocoformat.json --ext xml

# Check output
$ ls sample/ | grep bccd_test_cocoformat.json
bccd_test_cocoformat.json

# Check output
cut -f -4 -d , sample/bccd_test_cocoformat.json
{"images": [{"file_name": "BloodImage_00007.jpg", "height": 480, "width": 640, "id": "BloodImage_00007"}
```

python voc2coco.py --ann_dir D:\\python pro\\pythonProject\\bloodcelld_test_project\\BCCD_Dataset\\BCCD\\Annotations --ann_ids D:\\python pro\\pythonProject\\bloodcelld_test_project\\BCCD_Dataset\\BCCD\\ImageSets\\Main\\train.txt --labels "red blood cell" "white blood cell" "platelet" --output D:\\python pro\\pythonProject\\bloodcelld_test_project\\YOLOX\\datasets\\BCCD\\annotations\\instances_train2017.json --ext xml --img_dir D:\\python pro\\pythonProject\\bloodcelld_test_project\\BCCD_Dataset\\BCCD\\JPEGImages

python voc2coco.py --ann_dir ../BCCD_Dataset/BCCD/Annotations --ann_ids ../BCCD_Dataset/BCCD/ImageSets/Main/train.txt --labels ../BCCD_Dataset/labels.txt --output ../YOLOX/datasets/BCCD/annotations/instances_train2017.json --ext xml
--img_dir ../BCCD_Dataset/JPEGImages

python voc2coco.py --ann_dir ../BCCD_Dataset/BCCD/Annotations --ann_ids ../BCCD_Dataset/BCCD/ImageSets/Main/val.txt --labels ../BCCD_Dataset/labels.txt --output ../YOLOX/datasets/BCCD/annotations/instances_val2017.json --ext xml

mkdir -p ../YOLOX/datasets/BCCD/images/train2017
mkdir -p ../YOLOX/datasets/BCCD/images/val2017

# 读取ID列表并复制文件
xargs -a ../BCCD_Dataset/ImageSets/Main/train.txt -I{} cp ../BCCD_Dataset/JPEGImages/{}.jpg ../YOLOX/datasets/BCCD/images/train2017/
xargs -a ../BCCD_Dataset/ImageSets/Main/val.txt -I{} cp ../BCCD_Dataset/JPEGImages/{}.jpg ../YOLOX/datasets/BCCD/images/val2017/

while read p; do 
    cp JPEGImages/$p.jpg ../YOLOX/datasets/BCCD/images/train2017/;
done <ImageSets/Main/train.txt

Get-Content BCCD_Dataset/BCCD/ImageSets/Main/train.txt | ForEach-Object {
    Copy-Item "BCCD_Dataset/BCCD/JPEGImages/$_.jpg" "../bloodcelld_test_project/YOLOX/datasets/BCCD/images/train2017/"
}

Get-Content BCCD_Dataset/BCCD/ImageSets/Main/val.txt | ForEach-Object {
    Copy-Item "BCCD_Dataset/BCCD/JPEGImages/$_.jpg" "../bloodcelld_test_project/YOLOX/datasets/BCCD/images/val2017/"
}

python -m yolox.tools.train -f exps/default/yolox_s.py -d 8 -b 64 --fp16 -o

cd YOLOX
pip install -v -e .
python tools/train.py -f exps/example/custom/yolox_bccd_nano_test.py -d 1 -b 4 --fp16 -o