import cv
import cv2
import numpy as np
import os
import glob
from six.moves import cPickle

stack = []

data_dictionary = {"data":[], "label":[]}

def pack_image_into_data(dir, label, stack):

	path = './data/' + dir + '/*.png'
	images = glob.glob(path)
	for img_name in images : 
		img = cv2.imread(img_name)
		resized_img = cv2.resize(img, (28, 28))
		image_gray = cv2.cvtColor(resized_img, cv2.cv.CV_BGR2GRAY)
		npimage = np.asarray(image_gray).reshape(1,784)[0]
		npimage = npimage/255.
		stack.append((npimage, label))
	return stack

stack = pack_image_into_data("negative", 0, stack)
stack = pack_image_into_data("positive", 1, stack)
stack = np.asarray(stack)

np.random.shuffle(stack)
stack = stack.tolist()
for t in stack :
	data_dictionary["data"].append(t[0])  	
	data_dictionary["label"].append(t[1])  	

with open('data.pkl', 'wb') as output:
  cPickle.dump(data_dictionary, output, -1)
  print data_dictionary["data"][0].shape
  print data_dictionary["label"][0]
  print "Saved ", len(data_dictionary["data"]), " images with ", len(data_dictionary["label"]), " labels."

			

