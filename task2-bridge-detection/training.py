from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os


def main():
    # 1. 注册coco数据集
    data_path = 'C:/Users/86159/desktop/competitions/gaofen/data2_bridge/data/'
    json_file = r'annotation.json'
    register_coco_instances(name="bridge_2000", metadata={}, image_root=data_path, json_file=json_file)

    # 2. 加载数据与模型
    cfg = get_cfg()
    cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("bridge_2000",)
    cfg.DATASETS.TEST = ()

    # 3. 自定义参数
    cfg.SOLVER.IMS_PER_BATCH = 3
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = 500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes (bridge)
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6  # 抑制效果显著
    cfg.MODEL.RPN.NMS_THRESH = 0.8
    cfg.OUTPUT_DIR = './output0731'

    # 4. 训练
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':
    setup_logger()
    main()
