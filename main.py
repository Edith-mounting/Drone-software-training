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
def convert_bounding_boxes_form_Yolo_Centroid_format(results):
	# clean rect so it is clean an can be filled with new detection from frame\
	# later used in conversion_to_x1y1x2y2 . Conversion from yolo format to Centroid Format
	# rects are needed for centroid to work. They need to be cleared every time
	rects = []

	# if len(results) <= 1: # check if array is not empty, for prevention of crashing in later stage
	#    return []
	# print("resultsinside convert bound..",results)
	try:
		for cat, score, bounds in results:  # unpacking
			# print("cat, score, bounds ", cat, score, bounds )
			x, y, w, h = bounds
			# print( "x, y, w, h", x, y, w, h)
			"""
            convert from yolo format to cetroid format
            Yolo output:
            [(b'person', 0.859128475189209, (243.3025360107422, 308.5773010253906, 183.75604248046875, 205.69090270996094))]
            [(b'person', 0.9299755096435547, (363.68475341796875, 348.0577087402344, 252.04286193847656, 231.17266845703125)), (b'vase', 0.3197628855705261, (120.3013687133789, 405.3641357421875, 40.76551055908203, 32.07142639160156))]
            [(b'mark', 0.9893345236778259, (86.11815643310547, 231.90643310546875, 22.100597381591797, 54.182857513427734)), (b'mark', 0.8441593050956726, (225.28382873535156, 234.5716094970703, 14.333066940307617, 53.428749084472656)), (b'edge', 0.6000953316688538, (377.6446838378906, 254.71759033203125, 8.562969207763672, 18.379894256591797)), (b'edge', 0.5561915636062622, (388.4414367675781, 211.0662841796875, 10.678437232971191, 15.206807136535645)), (b'edge', 0.44139474630355835, (377.0844421386719, 150.8873748779297, 9.128596305847168, 18.9124755859375)), (b'crack', 0.28897273540496826, (268.6462707519531, 169.00457763671875, 253.9573516845703, 34.764007568359375))]]
            centroid input: 
            [array([145, 153, 248, 274]), array([113, 178, 148, 224])]

            ([145, 153, 248, 274]), ([113, 178, 148, 224])
            """
			# calculate bounding box for every object from YOLO for centroid purposes
			box = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
			# box = ([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
			# append to list of  bounding boxes for centroid
			# rects.append(box.astype("int"))
			rects.append(box)
		return rects
	except Exception as e:
		print(e)
		print("There was a problem with extrection from result:", rects)
		return rects
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
def update_resutls_for_id(results,ct_objects):
    """
    loop over the tracked objects from Yolo34
    Reconstruct Yolo34 results with object id (data from centroid tracker) an put object ID to idresults list, like :
    class 'list'>[(b'person', 0.9972826838493347, (646.4600219726562, 442.1628112792969, 1113.6322021484375, 609.4992065429688)), (b'bottle', 0.5920438170433044, (315.3851318359375, 251.22744750976562, 298.9032287597656, 215.8708953857422))]
    class 'list'>[(1, b'person', 0.9972826838493347, (646.4600219726562, 442.1628112792969, 1113.6322021484375, 609.4992065429688)), (4, b'bottle', 0.5920438170433044, (315.3851318359375, 251.22744750976562, 298.9032287597656, 215.8708953857422))]
    :param results from Yolo34:
    :return:idresults
    """
    idresults = []
    for cat, score, bounds in results:
        x, y, w, h = bounds
        # chyba_one_mark_small("cell phone", cat, score, x, y, w, h, )
        # loop over the tracked objects from Centroid
        #print(ct_objects)
        for ct_object in ct_objects.items():
            # put centroid id and coordinates to cX and Cy variables
            id, coordinates = ct_object[0], ct_object[1]
            cX, cY = coordinates[0], coordinates[1]
            # there is difference between yolo34 centroids and centroids calculated by centroid tracker,Centroid closer then 2 pixel are considired to matcg  TODO find where?
            if abs(cX - int(x)) <= 2 and abs(cY - int(y)) <= 2:
                # reconstruct detection list as from yolo34 including ID from centroid
                idresult = id, cat, score, bounds
                idresults.append(idresult)
    return idresults
##############################################################################################################
'''def update_resutls_for_distance(detections):

    distance_results = []
    defined_categories = ["person", "cell phone"]
    for id,cat, score, bounds in detections:
        # print(cat)
        if "person" == bytes.decode(cat):
            dist_t_camera_p = distance_to_camera(7, know_distance(Xresolution), bounds[2])
            distance_result = dist_t_camera_p, id, cat, score, bounds

        if "cell phone" == bytes.decode(cat):
            dist_t_camera_c = distance_to_camera(5, know_distance(Xresolution), bounds[2])
            distance_result = dist_t_camera_c, id ,cat, score, bounds

        if not bytes.decode(cat) in defined_categories:
            #logging.info("Not defined cat using default fordistance calculation ")
            dist_t_camera_default = distance_to_camera(6, know_distance(Xresolution), bounds[2])
            distance_result = dist_t_camera_default, id, cat, score, bounds

        distance_results.append(distance_result)
        # print(distance_results)
    return  distance_results'''
##############################################################################################################
import cv2
import numpy as np
# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("detected_video.avi", fourcc , 25, (852, 480))

camera = cv2.VideoCapture("Subway - 6398.mp4")
ct = CentroidTracker(maxDisappeared=60)

while True:
    _,img = camera.read()
    img=cv2.resize(img,(1020,720))
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    ct_objects = ct.update(convert_bounding_boxes_form_Yolo_Centroid_format(outs))
    id_detections = update_resutls_for_id(outs, ct_objects)
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