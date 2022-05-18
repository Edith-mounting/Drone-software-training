# import the necessary packages
from collections import OrderedDict

import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker():
	def __init__(self, maxDisappeared=50):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			#for objectID in self.disappeared.keys():
			# below is fix of upper line found at https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/ at posts
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# return early as there are no centroids or tracking info
			# to update
			return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the bounding box rectangles
		#for (i, (cat, startX, startY, endX, endY)) in enumerate(rects):
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				# val
				if row in usedRows or col in usedCols:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# return the set of trackable objects
		return self.objects
##############################################################################################################
scale_trail_visualization = 4 # how much compact trail visualization heigth
cell_phone = []
list_chyba = []
# Used by pLoopTrigerlist  to communicate with main loop format is [(2.1, 1551338571.7396123, 2.2, 1551338571.9881353), (3.1, 1551338578.9405866, 3.2, 1551338579.1024451), (0.1, 1551338586.2836142, 0.2, 1551338586.4773874)]
trigerlist = []
idresults = []
# Used by pLoopTrigerlist  to confirm object was marked  format is [(2.1, 1551338571.7396123), (2.2, 1551338571.9881353), (3.1, 1551338578.9405866), (3.2, 1551338579.1024451), (0.1, 1551338586.2836142), (0.2, 1551338586.4773874)]
#fastTrigerList shall be deleted in future releasesdsa
alreadyBlinkedTriger =[]
alreadyBlinkedList = []
objekty = {}  # it is storing all detection from program startup

how_big_object_max_small = 0.9  # detect object from how_big_object_min_small to how_big_object_max_small size of screen
how_big_object_min_small = 0.05 # detect object from how_big_object_min_small to how_big_object_max_small size of screen
number_of_deleted_objects = 0 # used  for main is to be deleted
detection_treshold = 0.50     # percentage which detection to consider

#Colors
black=(0,0,0)
white = (255,255,255)
red = (0,0,255)
green = (0,255,0)
blue = (255, 0, 0)
aqua = (0,255,255)
fuchsia = (255,0,255)
maroon = (128,0,0)
navy = (0,0,128)
olive = (128,128,0)
purple = (128,0,128)
teal = (0,128,128)
yellow = (255,255,0)
azzure = (255, 255, 0)
brown = (19, 69,139)
magenta = (255, 0, 255)
orange =(0, 128, 255)

font_size =0.7              # used for drawing on screen
delay_off_whole_program = 0 # speed of whole program it gives delay to YOLO loop
max_Yobject = 50            # maxinimum amount of objects to keep im memory older than the mentioned number will be deleted
error_next_possible_blink_min = 50   # value in angles from magneto
second_next_possible_blink_min = 100 # value in angles from magneto


min_distance = 70

##############################################################################################################
def rotate_to_target (navigate_frame,detection):

    x, y, w, h = detection[3][0], \
                         detection[3][1], \
                         detection[3][2], \
                         detection[3][3]
    middleX= Xresolution / 2
    middleY= Yresolution / 2
    darknetvscameraresolutionx = (Xresolution / network_width)
    darknetvscameraresolutiony = (Yresolution / network_heigth)
    #x = x * darknetvscameraresolutionx
    #y = y * darknetvscameraresolutiony
    #w = w * darknetvscameraresolutionx
    #h = h * darknetvscameraresolutiony
    try:
        if x < middleX:
            logging.info("Rotate to left")
            #calculate how much to rotate for now 1=Left with max power,
            cv2.line(navigate_frame, (0, 100), (100, 100), green, 5)
            return 1
        if x > middleX:
            logging.info("Rotate to right")
            cv2.line(navigate_frame, (100, 100), (200, 100), green, 5)
            return -1
    except Exception as e:
        print(e)
##############################################################################################################
import cv2
import numpy as np
import tellopy
import av
# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading video
drone = tellopy.Tello()
drone.connect()
#drone.start_video()
drone.wait_for_connection(60.0)
#drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
#container = av.open(ns.drone.get_video_stream())
container = av.open(drone.get_video_stream())
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("detected_video.avi", fourcc , 25, (852, 480))
x = 200
y = 200
w = 200
h = 200
track_window = (x, y, w, h)
frame_skip=300
camera = cv2.VideoCapture(0)

while True:
	for frame in container.decode(video=0):
		if 0 < frame_skip:
			frame_skip = frame_skip - 1
			continue
		#start_time = time.time()
	img = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
	k = cv2.waitKey(1)
    #_,img = camera.read()
	img=cv2.resize(img,(1020,720))
	height, width, channels = img.shape

    # Detecting objects
	blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)

    # Showing informations on the screen
	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5:
				# Object detected
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)
                # Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
			cv2.putText(img, label, (x, y - 3), font, 2, color, 2)
  #  imgs=cv2.resize(img,(1020,720))
	cv2.imshow("Image", img)
	key = cv2.waitKey(1)
	if key == 27:
			break

camera.release()
cv2.destroyAllWindows()