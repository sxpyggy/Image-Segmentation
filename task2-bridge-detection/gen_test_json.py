"""生成test数据集的coco文件"""
import datetime
import json
import os
from PIL import Image
from pycococreatortools import pycococreatortools


'''============================== docker上改为/input_path ===================='''
data_path = './input_path'

INFO = {
    "year": 2020,
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

CATEGORIES = [
    {
        'id': 1,
        'name': 'bridge',  # 桥梁，只有一类
        'supercategory': 'bridge',
    }
]


def main():
    # 1. 定义最终json文件字典
    print("======== 1. 定义最终json文件字典 ========")
    coco_output = {
        "info": INFO,
        "categories": CATEGORIES,
        "images": []
    }

    # 2. 提取images信息
    print("======== 2. 提取images信息 ========")
    tif_files = os.listdir(data_path)
    image_id = 1

    for i in range(len(tif_files)):
        img = data_path + '/' + tif_files[i]
        image = Image.open(img)

        image_name = os.path.basename(img)
        image_info = pycococreatortools.create_image_info(
            image_id, image_name, image.size)
        coco_output["images"].append(image_info)

        image_id += 1

    # 3. 字典写入json文件
    print("======== 3. 字典写入json文件 ========")
    with open('annotation_test.json', 'w') as output_json_file:
        json.dump(coco_output, output_json_file, indent=4)


if __name__ == '__main__':
    main()
