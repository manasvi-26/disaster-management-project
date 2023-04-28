import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import csv
import json
import datetime
from math import floor



def get_crowd_data():

	human_count = []
	abnormal_activity = []

	with open('../processed_data/crowd_data.csv', 'r') as file:
		reader = csv.reader(file, delimiter=',')
		next(reader)
		for row in reader:
			human_count.append(int(row[1]))
			abnormal_activity.append(bool(int(row[2])))
	
	return human_count, abnormal_activity

def get_video_data(human_count):
	with open('../processed_data/video_data.json', 'r') as file:
		data = json.load(file)
		data_record_frame = data["DATA_RECORD_FRAME"]
		is_cam = data["IS_CAM"]
		vid_fps = data["VID_FPS"]
		start_time = data["START_TIME"]

	start_time= datetime.datetime.strptime(start_time, "%d/%m/%Y, %H:%M:%S")
	time_steps = data_record_frame/vid_fps
	data_length = len(human_count)
	return start_time, time_steps, data_length

def plot(human_count, abnormal_activity,start_time, time_steps, data_length):

	time_axis = []
	graph_height = max(human_count)

	fig, ax = plt.subplots()
	time = start_time
	for i in range(data_length):
		time += datetime.timedelta(seconds= time_steps)
		time_axis.append(time)
		next_time = time + datetime.timedelta(seconds= time_steps)
		rect_width = mdates.date2num(next_time) - mdates.date2num(time)
		if abnormal_activity[i]:
			ax.add_patch(patches.Rectangle((mdates.date2num(time), 0), rect_width, graph_height / 20, facecolor = 'blue', fill=True))


	crowd_line,=  plt.plot(time_axis, human_count, linewidth=3, label="Crowd Count")
	plt.title("Crowd Data versus Time")
	plt.xlabel("Time")
	plt.ylabel("Count")
	an_legend = patches.Patch(color= "blue", label="Abnormal Crowd Activity Detected")
	plt.legend(handles=[crowd_line, an_legend])
	plt.show()	

def main():

	human_count, abnormal_activity =get_crowd_data()
	start_time, time_steps, data_length = get_video_data(human_count)
	plot(human_count, abnormal_activity,start_time, time_steps, data_length)

main()