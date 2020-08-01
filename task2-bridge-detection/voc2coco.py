# Pascal voc转COCO
import datetime
import json
import os
from PIL import Image
from pycococreatortools import pycococreatortools
import xml.etree.ElementTree as ET


data_path = 'C:/Users/86159/desktop/competitions/gaofen/data2_bridge/data/'

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


def compute_polygon_area(points):
    """计算多边形面积--暂时不用"""
    point_num = len(points)
    if point_num < 3:
        return 0.0
    s = points[0][1] * (points[point_num - 1][0] - points[1][0])
    for i in range(1, point_num):
        s += points[i][1] * (points[i - 1][0] - points[(i + 1) % point_num][0])
    return abs(s / 2.0)


def extract_contours(xml_file):
    """提取xml文件中的轮廓多边形坐标"""
    # 1. 解析xml文件，找到objects分支
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = root.find('objects')

    # 2. 从xml中提取多个目标名和轮廓坐标
    contours = []
    for object in objects:
        contour = []
        points = object.find('points').findall('point')
        for point in points:
            contour.append(eval(point.text))
        contours.append(contour)

    # 3.返回多个轮廓坐标
    return contours


def voc2coco():
    # 1. 定义最终json文件字典
    print("======== 1. 定义最终json文件字典 ========")
    coco_output = {
        "info": INFO,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    # 2. 分离xml和tif文件
    print("======== 2. 分离xml和tif文件 ========")
    files = os.listdir(data_path)
    tif_files = []
    xml_files = []

    for name in files:
        if name.endswith(".tif"):
            tif_files.append(name)
        elif name.endswith(".xml"):
            xml_files.append(name)
    tif_files.sort()
    xml_files.sort()

    # 3. 提取images和segmentations信息
    print("======== 3. 提取images和segmentations信息 ========")
    image_id = 1
    segmentation_id = 1
    for i in range(len(tif_files)):
        # 3.1 提取image信息
        img = data_path + '/' + tif_files[i]
        image = Image.open(img)

        image_name = os.path.basename(img)
        image_info = pycococreatortools.create_image_info(
            image_id, image_name, image.size)

        coco_output["images"].append(image_info)

        # 3.2 提取segmentations信息
        xml = data_path + '/' + xml_files[i]
        contours = extract_contours(xml)

        for contour in contours:
            bbox = []
            bbox.extend(contour[3])  # 以第四个为左上角
            bbox.extend(contour[1])  # 以第四个为右下角

            # 将一个bbox加入字典
            annotation_info = {'id': segmentation_id, 'image_id': image_id, 'category_id': 1,
                               'is_crowd': 0, 'bbox': bbox}

            coco_output["annotations"].append(annotation_info)

            segmentation_id += 1

        image_id += 1

    # 4. 字典写入json文件
    print("======== 4. 字典写入json文件 ========")
    with open('annotation.json', 'w') as output_json_file:
        json.dump(coco_output, output_json_file, indent=4)


if __name__ == '__main__':
    voc2coco()
