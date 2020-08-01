# -*- coding: utf-8 -*-
"""标注ground truth并查看"""
import xml.etree.ElementTree as ET
import cv2
import numpy as np


def extract_contours(xml_file):
    """提取xml文件中的轮廓多边形坐标"""
    # 1. 解析xml文件，找到objects分支
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = root.find('objects')

    # 2. 从xml中提取多个目标名和轮廓坐标
    contours = []
    for object in objects:
        # object_name = object.find('possibleresult').find('name').text
        contour = []
        points = object.find('points').findall('point')
        for point in points:
            contour.append(eval(point.text))
        contours.append(contour)

    # 3.返回多个轮廓坐标
    return contours


def draw_contours(img_file, contours):
    """在图片上标注轮廓"""
    im = cv2.imread(img_file)
    contours = np.array(contours).astype('int')
    # print(contours)
    cv2.drawContours(im, contours, -1, (255, 255, 255))  # -1同时绘制多个轮廓
    return im


def main():
    path = 'C:/Users/86159/desktop/competitions/gaofen/data2_bridge/data/'
    num = 1037
    # 显示gt
    xml_file = path + str(num) + '.xml'
    img_file = path + str(num) + '.tif'

    contours = extract_contours(xml_file)
    im = draw_contours(img_file, contours)

    cv2.imshow(img_file, im)
    cv2.waitKey(0)

    # 显示预测
    xml_file = "./output_path/" + str(num) + ".xml"
    img_file = path + str(num) + '.tif'

    contours = extract_contours(xml_file)
    im = draw_contours(img_file, contours)

    cv2.imshow(img_file, im)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
