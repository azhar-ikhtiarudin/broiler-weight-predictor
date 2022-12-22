from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.config import CfgNode as CN 


def add_broilercv_config(cfg: CN):
    """Add broilercv specific config."""
    cfg.INFO = CN()
    cfg.INFO.DESCRIPTION = "Broiler instance segmentation COCO dataset"
    cfg.INFO.VERSION = "1.0.0"
    cfg.INFO.CONTRIBUTOR = "TF"
    cfg.INFO.LABEL = "broiler"
    cfg.INFO.PERIOD_START = "2021-11-24"
    
    cfg.FRAME_DIMENSION = (1080, 1920)
    cfg.CROP_DIMENSION = ((0, 1030), (254, 1664))

    cfg.AREA = CN()
    # Reference object length for camera calibration
    cfg.AREA.REFERENCE_MM = 51
    cfg.AREA.REFERENCE_PX = 45

    cfg.WEIGHT = CN()
    cfg.WEIGHT.THETA = (-3.448, 2.086, -0.304)  # (theta0, theta1, theta2)

    cfg.TRACKER = CN()
    cfg.TRACKER.IOU_THRESHOLD = 0.5
    cfg.TRACKER.MIN_BOX_REL_DIM = 0.02
    cfg.TRACKER.MIN_BOX_REL_EDGE = 0.0
    cfg.TRACKER.MAX_FRAME_LOST_COUNT = 8

    cfg.DATASETS.TRAIN = ("broilercv",)
    cfg.DATASETS.TEST = ("broilercv-val",)
    cfg.TEST.EVAL_PERIOD = 20
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.MASK_ON = True
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 600    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    cfg.MODEL.WEIGHTS = ("model/model_final.pth")


register_coco_instances("broilercv", {}, "data/train/annotations.json", "data/train/")
register_coco_instances("broilercv-val", {}, "data/val/annotations.json", "data/val/")
