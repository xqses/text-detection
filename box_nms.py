# import the necessary packages
import numpy as np
# Malisiewicz et al.


def box_nms(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes
	pick = []
	# grab the coordinates of the bounding boxes
	# print(boxes)
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	# print(area)
	idxs = np.argsort(y2)
	# print(idxs)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes

		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]

		# last = len(idxs) - 1
		# print(last, "last")
		# i = idxs[last]
		# print(i, "idx")
		# pick.append(i)


		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		# xx1 = np.maximum(x1[i], x1[idxs[:last]])
		# print(xx1, "xx1")
		# yy1 = np.maximum(y1[i], y1[idxs[:last]])
		# print(yy1, "yy1")
		# xx2 = np.minimum(x2[i], x2[idxs[:last]])
		# print(xx2, "xx2")
		# yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# print(yy2, "yy2")
		# compute the width and height of the bounding box
		# w = np.maximum(0, xx2 - xx1 + 1)
		# print(w, "w")
		# h = np.maximum(0, yy2 - yy1 + 1
		# # compute the ratio of overlap
		# overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		# idxs = np.delete(idxs, np.concatenate(([last],
		# 	np.where(overlap > overlapThresh)[0])))

		# loop over all indexes in the indexes list
		for pos in range(0, last):
			# grab the current index
			j = idxs[pos]
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			# print(xx1)
			yy1 = max(y1[i], y1[j])
			# print(yy1)
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]
			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)
		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)
	# return only the bounding boxes that were picked
	return np.array(boxes[pick]).astype("int")
	# return only the bounding boxes that were picked using the
	# integer data type