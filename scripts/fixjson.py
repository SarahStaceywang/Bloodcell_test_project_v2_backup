import json

import re
input_path = "/media/sata4/hzh/bccd/Bloodcell_test_project_v2/YOLOX/datasets/BCCD/annotations/instances_val2017.json"

output_path = "/media/sata4/hzh/bccd/Bloodcell_test_project_v2/YOLOX/datasets/BCCD/annotations/instances_val2017_fixed.json"


with open(input_path, "r") as f:
    data = json.load(f)

# 构建一个 mapping：filename_id(str) -> integer_id
image_id_map = {}
for i, img in enumerate(data["images"]):
    # 分配新的整数ID
    new_id = i + 1
    image_id_map[img["id"]] = new_id  # img["id"] 是原来的字符串
    img["id"] = new_id                # 替换为整数

# 修正 annotation 的 image_id
for anno in data["annotations"]:
    original_id = anno["image_id"]
    anno["image_id"] = image_id_map[original_id]

with open(output_path, "w") as f:
    json.dump(data, f)

print("✅ 修复完成！文件已保存为：", output_path)