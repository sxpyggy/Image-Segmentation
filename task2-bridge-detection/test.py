from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.catalog import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import cv2
import re
import os
from xml.dom.minidom import Document


"""=================== 生成xml文件的函数 ===================="""


def gen_info(parent_node, child_node_name, node_text):
    """生成一行叶子行"""
    # 如 <research> / <version>4.0</version>
    doc = Document()
    child_node = doc.createElement(child_node_name)
    parent_node.appendChild(child_node)
    child_node.appendChild(doc.createTextNode(node_text))


def gen_source(doc, annotation, im_name):
    """生成source部分"""
    source = doc.createElement("source")
    annotation.appendChild(source)
    gen_info(source, 'filename', im_name)  # 变量1
    gen_info(source, 'origin', "GF2/GF3")


def gen_research(doc, annotation):
    """生成research部分"""
    research = doc.createElement("research")
    annotation.appendChild(research)

    gen_info(research, "version", "4.0")
    gen_info(research, "provider", "中国人民大学统计学院")
    gen_info(research, "author", "高光远，杨睿")
    research.appendChild(doc.createComment("参赛课题"))
    gen_info(research, "pluginname", "桥梁目标识别")
    gen_info(research, "pluginclass", "识别")
    gen_info(research, "time", "2020-07-2020-11")


def gen_objects(doc, annotation, pre_result):
    """生成objects部分"""
    # 参数pre: 预测结果redictor(img)的输出
    annotation.appendChild(doc.createComment("存放目标检测信息"))
    objects = doc.createElement("objects")
    annotation.appendChild(objects)

    pred_boxes = pre_result["instances"].get("pred_boxes")
    scores = pre_result["instances"].get("scores")
    pred_classes = pre_result["instances"].get("pred_classes")
    classes = {0: "bridge"}

    for i in range(len(pred_boxes)):
        object = doc.createElement("object")
        objects.appendChild(object)
        gen_info(object, "coordinate", "pixel")
        gen_info(object, "type", "rectangle")
        gen_info(object, "description", "None")
        possibleresult = doc.createElement("possibleresult")
        object.appendChild(possibleresult)
        gen_info(possibleresult, "name", classes[pred_classes[i].item()])  # 变量3，提取
        gen_info(possibleresult, "probability", str(round(scores[i].item(), 2)))
        object.appendChild(doc.createComment("检测框坐标，首尾闭合的矩形，起始点无要求"))

        # 关键部分：坐标
        points = doc.createElement("points")
        object.appendChild(points)

        for coordinate in pred_boxes[i]:
            # 只有一个coordinate: tensor([x1, y1, x2, y2], device='cuda:0')
            x1 = round(coordinate[0].item(), 6)
            y1 = round(coordinate[1].item(), 6)
            x2 = round(coordinate[2].item(), 6)
            y2 = round(coordinate[3].item(), 6)
            """======================================手动更改右下角预测太多的情况========================================="""
            x2 = 0.5 * (x1 + x2)
            y2 = 0.5 * (y1 + y2)

            gen_info(points, "point", str(x1) + ', ' + str(y2))  # (x1, y2)
            gen_info(points, "point", str(x2) + ', ' + str(y2))  # (x2, y2)
            gen_info(points, "point", str(x2) + ', ' + str(y1))  # (x2, y1)
            gen_info(points, "point", str(x1) + ', ' + str(y1))  # (x1, y1)
            gen_info(points, "point", str(x1) + ', ' + str(y2))  # (x1, y2)


def gen_xml(im_name, pre_result, output_dir):
    # 解析结果，生成xml文件
    doc = Document()
    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)
    gen_source(doc, annotation, im_name)
    gen_research(doc, annotation)
    gen_objects(doc, annotation, pre_result)

    f = open(output_dir + im_name[:-4] + ".xml", "w", encoding='utf-8')
    f.write(doc.toprettyxml(indent=" "))
    f.close()


def main():
    # 1. 读取与注册数据集
    print("============ 1. 读取与注册数据集 =============")
    test_data = 'C:/Users/86159/desktop/competitions/gaofen/data2_bridge/data/'
    test_json = r'annotation_test.json'
    register_coco_instances(name="bridge_test50", metadata={}, image_root=test_data, json_file=test_json)
    dataset_dicts = DatasetCatalog.get("bridge_test50")
    test_metadata = MetadataCatalog.get("bridge_test50")

    # 2. 调用模型进行预测
    print("============ 2. 调用模型进行预测 ============")
    cfg = get_cfg()
    cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "./output0731_2/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # 设置测试门槛值
    cfg.DATASETS.TEST = ("bridge_test50",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes (bridge)
    predictor = DefaultPredictor(cfg)

    # 3. 展示与保存预测结果
    print("============ 3. 展示与保存预测结果 ============")
    output_dir = './output_path/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for data in dataset_dicts:
        img = cv2.imread(data["file_name"])
        # 3.1 提取文件名
        im_name = re.match(r'(.*)/data/(.*)', data["file_name"]).group(2)
        # 3.2 获取预测结果
        pre_result = predictor(img)
        # 3.3 将结果转为xml
        gen_xml(im_name, pre_result, output_dir)

        # # 3.4 显示标注结果
        # # ColorMode为"IMAGE_BW"移除未分割的像素颜色, 选择"IMAGE"保留原颜色
        # v = Visualizer(img[:, :, ::-1], metadata=test_metadata, scale=0.6, instance_mode=ColorMode.IMAGE)
        # pre = v.draw_instance_predictions(pre_result["instances"].to('cpu'))
        # img = pre.get_image()[:, :, ::-1]
        # cv2.imshow(im_name, img)
        # cv2.waitKey(0)


if __name__ == '__main__':
    main()
