from libs import detection
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os


USE_GPU = False
MIN_DISTANCE = 50

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = r'C:\Users\Asus\Documents\Projects\AI-IOT\yolo-coco\yolov3\coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = r'C:\Users\Asus\Documents\Projects\AI-IOT\yolo-coco\yolov3\yolov3.weights'
configPath = r'C:\Users\Asus\Documents\Projects\AI-IOT\yolo-coco\yolov3\yolov3.cfg'

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
fps = int(vs.get(cv2.CAP_PROP_FPS))

# loop over the frames from the video stream
iframe, avg_violate_per_second = 0, 0
Nviolate, Nresults, perc = [], [], [0,0]
iframe = 0
# initialize the set of indexes that violate the minimum social distance
violate = set()
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# resize the frame and then detect people (and only people) in it
	frame = imutils.resize(frame, width=700)

	if iframe%1 == 0:
		iframe = 0

		objects = ['person']
		results = detection.detect_object(frame, net, ln, Idxs=[LABELS.index(i) for i in objects if LABELS.index(i) is not None])

		# initialize the set of indexes that violate the minimum social distance
		violate = set()

		# ensure there are at least two people detections
		if len(results) >= 2:
			# extract all centroids from the results
			centroids = np.array([r[3] for r in results])
			# get the widths of bounding boxes
			dXs = [r[2][2]-r[2][0] for r in results]

			for i in range(len(results)):
				c1 = centroids[i]
				for j in range(i + 1, len(results)):
					c2 = centroids[j]
					Dx, Dy = np.sqrt((c2[0]-c1[0])**2), np.sqrt((c2[1]-c1[1])**2)
					thresX = (dXs[i] + dXs[j]) * 0.7
					thresY = (dXs[i] + dXs[j]) * 0.25
					# check to see if the distance between any pairs is less than the threshold
					if Dx<thresX and Dy<thresY:
						# update our violation set with the indexes of the centroid pairs
						violate.add(i)
						violate.add(j)

		# loop over the results
		for (i, (classID, prob, bbox, centroid)) in enumerate(results):
			# extract the bounding box and centroid coordinates, then
			# initialize the color of the annotation
			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			color = (0, 255, 0)

			# if the index pair exists within the violation set, then update the color
			if i in violate:
				color = (0, 0, 255)

			# draw (1) a bounding box around the person and (2) the centroid coordinates of the person,
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.circle(frame, (cX, cY), 5, color, 1)

	# draw the total number of social distancing violations on the output frame
	perc[0] = len(violate) / len(results) * 100
	Nviolate.append(len(violate))
	Nresults.append(len(results))
	if iframe > 0 and iframe%fps == 0:
		tmp, N = 0, 0
		for ii in range(iframe,iframe-(fps-1),-1):
			tmp += Nviolate[ii]
			N += Nresults[ii]
		avg_violate_per_second = tmp / fps
		perc[1] = avg_violate_per_second / N * fps * 100


	info = [
		("Jumlah Pelanggaran", len(violate)),
		("Avg Pelanggaran Per Detik", avg_violate_per_second),
	]

	rectangle_bgr = (255, 0, 0)
	# make the coords of the box with a small padding of two pixels
	overlay = frame.copy()
	cv2.rectangle(overlay, (5,330), (550,385), rectangle_bgr, -1)
	# opacity
	alpha=0.45

	# loop over the info tuples and draw them on our frame
	font_scale = .8
	font = cv2.FONT_ITALIC
	for (i, (k, v)) in enumerate(info):
		text = "{}: {} ({}%)".format(k, round(v,1), np.round(perc[i],1))
		cv2.putText(overlay, text, (10, frame.shape[0] + ((i * 25) - 40)),
			cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
	cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


	# check to see if the output frame should be displayed to our screen
	if args["display"] > 0:
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# if an output video file path has been supplied and the video
	# writer has not been initialized, do so now
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, fps, (frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output video file
	if writer is not None:
		writer.write(frame)

	iframe += 1