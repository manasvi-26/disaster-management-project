import numpy as np
import cv2
from utils.config import MIN_CONF, NMS_THRESH 

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

def detect_human (net, ln, frame, encoder, tracker, time):
	(frame_height, frame_width) = frame.shape[:2]
	boxes = []
	centroids = []
	confidences = []

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)

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
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	tracked_bboxes = []
	expired = []
	if len(idxs) > 0:
		del_idxs = []
		for i in range(len(boxes)):
			if i not in idxs:
				del_idxs.append(i)
		for i in sorted(del_idxs, reverse=True):
			del boxes[i]
			del centroids[i]
			del confidences[i]

		boxes = np.array(boxes)
		centroids = np.array(centroids)
		confidences = np.array(confidences)
		features = np.array(encoder(frame, boxes))
		detections = [Detection(bbox, score, centroid, feature) for bbox, score, centroid, feature in zip(boxes, scores, centroids, features)]

		tracker.predict()
		expired = tracker.update(detections, time)


		for track in tracker.tracks:
				if not track.is_confirmed() or track.time_since_update > 5:
						continue 
				tracked_bboxes.append(track)
	print(tracked_bboxes,expired)
	
	return [tracked_bboxes, expired]

