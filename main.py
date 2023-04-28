import datetime
import time
import numpy as np
import imutils
import cv2
import os
import csv
import json
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from utils.config import *
from video_process import video_process


def load_yolo_model():
    WEIGHTS_PATH = YOLO_CONFIG["WEIGHTS_PATH"]
    CONFIG_PATH = YOLO_CONFIG["CONFIG_PATH"]

    net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, ln


def initialize_tracker():
    max_cosine_distance = 0.7
    nn_budget = None

    max_age = DATA_RECORD_RATE * TRACK_MAX_AGE
    if max_age > 30:
        max_age = 30

    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=max_age)

    return encoder, tracker

def create_csv_files():
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')

    movement_data_file = open('processed_data/movement_data.csv', 'w')
    crowd_data_file = open('processed_data/crowd_data.csv', 'w')

    movement_data_writer = csv.writer(movement_data_file)
    crowd_data_writer = csv.writer(crowd_data_file)

    if os.path.getsize('processed_data/movement_data.csv') == 0:
        movement_data_writer.writerow(
            ['Track ID', 'Entry time', 'Exit Time', 'Movement Tracks'])

    if os.path.getsize('processed_data/crowd_data.csv') == 0:
        crowd_data_writer.writerow(
		    ['Time', 'Human Count', 'Abnormal Activity'])
    

    return movement_data_writer,crowd_data_writer


def main():
    encoder, tracker = initialize_tracker()
    net, ln = load_yolo_model()
    cap = cv2.VideoCapture(VIDEO_CONFIG["VIDEO_CAP"])
    movement_data_writer,crowd_data_writer = create_csv_files()

    video_process(cap, FRAME_SIZE, net, ln, encoder, tracker, movement_data_writer, crowd_data_writer)
    cv2.destroyAllWindows()
    
    cap.release()

main()
