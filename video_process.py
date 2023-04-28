import time
import datetime
import numpy as np
import imutils
import cv2
import time
from math import ceil
from scipy.spatial.distance import euclidean
from tracking import detect_human
from utils.util import rect_distance, progress, kinetic_energy
from utils.colors import RGB_COLORS
from utils.config import *
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet


def _record_movement_data(movement_data_writer, movement):
	track_id = movement.track_id 
	entry_time = movement.entry 
	exit_time = movement.exit			
	positions = movement.positions
	positions = np.array(positions).flatten()
	positions = list(positions)
	data = [track_id] + [entry_time] + [exit_time] + positions
	movement_data_writer.writerow(data)

def _record_crowd_data(time, human_count, abnormal_activity, crowd_data_writer):
	data = [time, human_count,int(abnormal_activity)]
	crowd_data_writer.writerow(data)

def _end_video(tracker, frame_count, movement_data_writer):
	for t in tracker.tracks:
		if t.is_confirmed():
			t.exit = frame_count
			_record_movement_data(movement_data_writer, t)



def video_process(cap, frame_size, net, ln, encoder, tracker, movement_data_writer, crowd_data_writer):
    
    frame_count = 0

    VID_FPS = cap.get(cv2.CAP_PROP_FPS)
    DATA_RECORD_FRAME = int(VID_FPS / DATA_RECORD_RATE)
    TIME_STEP = DATA_RECORD_FRAME/VID_FPS

    while True:
        (ret, frame) = cap.read()
    
        if not ret:
            _end_video(tracker, frame_count, movement_data_writer)
            break


        frame_count += 1
        if frame_count % DATA_RECORD_FRAME != 0:
            continue

        frame = imutils.resize(frame, width=frame_size)
        
        [humans_detected, expired] = detect_human(net, ln, frame, encoder, tracker, frame_count)

        for movement in expired:
            _record_movement_data(movement_data_writer, movement)
		
        abnormal_individual = []
        ABNORMAL = False

        for i, track in enumerate(humans_detected):
            [x, y, w, h] = list(map(int, track.to_tlbr().tolist()))
            [cx, cy] = list(map(int, track.positions[-1]))
            idx = track.track_id

            ke = kinetic_energy(track.positions[-1], track.positions[-2], TIME_STEP)
            if ke > ABNORMAL_ENERGY:
                abnormal_individual.append(track.track_id)
            
            cv2.rectangle(frame, (x, y), (w, h), RGB_COLORS["green"], 2)
            cv2.putText(frame, str(int(idx)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RGB_COLORS["green"], 2)

            if len(humans_detected)  > ABNORMAL_MIN_PEOPLE:
                if len(abnormal_individual) / len(humans_detected) > ABNORMAL_THRESH: ABNORMAL = True

            
        if ABNORMAL_CHECK:
            if ABNORMAL:
                for track in humans_detected:
                    if track.track_id in abnormal_individual:
                        [x, y, w, h] = list(map(int, track.to_tlbr().tolist()))
                        cv2.rectangle(frame, (x , y ), (w, h), RGB_COLORS["blue"], 5)
        
        text = "Crowd count: {}".format(len(humans_detected))
        cv2.putText(frame, text, (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        
        _record_crowd_data(frame_count, len(humans_detected),  ABNORMAL, crowd_data_writer)

        cv2.imshow("Processed Output", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            _end_video(tracker, frame_count, movement_data_writer)
            break
    
    cv2.destroyAllWindows()