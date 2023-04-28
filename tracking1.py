import numpy as np
import cv2
from utils.config import MIN_CONF, NMS_THRESH 
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

def detect_human(net, ln, frame, encoder, tracker, time):
    boxes, centroids, confidences = get_detections(net, ln, frame)
    idxs = non_max_suppression(boxes, confidences)

    if len(idxs) > 0:
        boxes, centroids, confidences = filter_detections(idxs, boxes, centroids, confidences)
        features = get_features(encoder, frame, boxes)
        detections = create_detections(boxes, confidences, centroids, features)
        expired_tracks = tracker.update(detections, time)
        tracked_bboxes = get_tracked_bboxes(tracker)

        return tracked_bboxes, expired_tracks
    else:
        return [], []

def get_detections(net, ln, frame):
    (frame_height, frame_width) = frame.shape[:2]
    boxes = []
    centroids = []
    confidences = []

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > MIN_CONF:
                box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                (center_x, center_y, width, height) = box.astype("int")
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                centroids.append((center_x, center_y))
                confidences.append(float(confidence))

    return boxes, centroids, confidences

def non_max_suppression(boxes, confidences):
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
    return idxs.flatten()

def filter_detections(idxs, boxes, centroids, confidences):
    boxes = np.array([boxes[i] for i in idxs])
    centroids = np.array([centroids[i] for i in idxs])
    confidences = np.array([confidences[i] for i in idxs])
    return boxes, centroids, confidences

def get_features(encoder, frame, boxes):
    return np.array(encoder(frame, boxes))

def create_detections(boxes, confidences, centroids, features):
    detections = [Detection(bbox, score, centroid, feature) for bbox, score, centroid, feature in zip(boxes, confidences, centroids, features)]
    return detections

def get_tracked_bboxes(tracker):
    tracked_bboxes = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 5:
            continue 
        tracked_bboxes.append(track)
    return tracked_bboxes
