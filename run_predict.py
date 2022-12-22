from src.setup import add_broilercv_config
from src.scripts.weight_predictor import Predictor
from src.scripts.weight_calculator import calculateWeight 
from detectron2.config import CfgNode as CN, get_cfg
import numpy as np
import json

from src.setup_mqtt import connect_mqtt, topic
from src.setup_sql import setup_sql
from src.scripts.tools import unixToDatetime, saveLocal, sendWeight

def init():
    cfg = get_cfg()
    cfg.merge_from_file("src/data/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    add_broilercv_config(cfg)
    return Predictor(cfg) # instances hasil class Predictor  


###MAIN PROGRAM###
#Parameter
video_src = np.array(['./data-test/video/1.mp4', './data-test/video/2.mp4', './data-test/video/3.mp4', './data-test/video/4.mp4', './data-test/video/5.mp4', './data-test/video/6.mp4', './data-test/video/7.mp4', './data-test/video/8.mp4', './data-test/video/9.mp4', './data-test/video/10.mp4', './data-test/video/11.mp4', './data-test/video/12.mp4', './data-test/video/13.mp4']) 
age_list = [11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24]
percent = 0.8
n_data = 3
predictor = init()
weightData = np.zeros((len(video_src),n_data))

total_average = calculateWeight(age_list, video_src, percent, n_data, weightData, predictor)
