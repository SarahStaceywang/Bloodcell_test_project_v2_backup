import json
import os

json_files = ["YOLOX/datasets/BCCD/annotations/instances_train2017_fixed_copy.json",
              "YOLOX/datasets/BCCD/annotations/fixed_instances_val2017_fixed_copy.json"]

for file_path in json_files:
    with open(file_path, 'r') as f:
        data = json.load(f)

    modified = False

    if "info" not in data:
        data["info"] = {
            "description": "BCCD Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "Sarah Stacey",
            "date_created": "2025-06-08"
        }
        modified = True

    if "licenses" not in data:
        data["licenses"] = [{
            "id": 1,
            "name": "Unknown",
            "url": "http://example.com"
        }]
        modified = True

    if modified:
        with open(file_path, 'w') as f:
            json.dump(data, f)
        print(f"✅ Fixed and saved: {file_path}")
    else:
        print(f"✅ Already OK: {file_path}")