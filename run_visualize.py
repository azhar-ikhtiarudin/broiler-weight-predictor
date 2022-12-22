from src.setup import add_broilercv_config
from src.scripts.weight_predictor import Predictor
# from configs.scripts.weight_calculator import calculateWeight, calculateWeightDummy
from src.scripts.weight_visualizer import WeightVisualizer
from detectron2.config import CfgNode as CN, get_cfg
# from paho.mqtt import client as mqtt_client
import numpy as np
import json
import cv2

# from configs.setup_mqtt import connect_mqtt, topic
# from configs.setup_sql import setup_sql
# from src.scripts.tools import unixToDatetime, saveLocal, sendWeight

def init():
    cfg = get_cfg()
    cfg.merge_from_file("src/data/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    add_broilercv_config(cfg)
    return Predictor(cfg) # instances hasil class Predictor  

age = 30
percent = 0.6

predictor = init()
im = cv2.imread("data-test/image/test-sample-1.jpg")
outputs = predictor(im, age, percent)
v = WeightVisualizer(im[:, :, ::-1],
                # metadata=broilercv_val_metadata, 
                scale=0.8, 
                age=f'{age}'
)
instances = outputs['instances']
detections = instances[instances.scores > percent]

w = v.draw_instance_predictions(instances.to("cpu"))
u = v.draw_weight_predictions(outputs)

while True:
    cv2.imshow('image-drawing', u.get_image()[:, :, ::-1])
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()