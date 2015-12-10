import cv
import cv2
import random 
import numpy as np

img = cv2.imread('./book/1.jpg')
width, height, channels = img.shape
img = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)

wsize = 28
gennum = 16000

np.asarray(img)

for i in range(gennum):
  x = random.randint(0,width - wsize)
  y = random.randint(0,height - wsize)
  cropped_img = img[x:x+wsize, y:y+wsize]
  filename = "./data/negative/"+str(i)+".png"
  cv2.imwrite(filename, cropped_img)

