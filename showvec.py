import struct,array
import os
import cv2
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser(
    description='A Converter from Vector to Images')
parser.add_argument('--img', '-i', default='', help='path of input image')
parser.add_argument('--folder', '-f', default='', help='name of parent folder')
args = parser.parse_args()

def showvec(fn, width=32, height=32, resize=1.0):
  f = open(fn,'rb')
  HEADERTYP = '<iihh' # img count, img size, min, max

  # read header
  imgcount,imgsize,_,_ = struct.unpack(HEADERTYP, f.read(12))

  for i in range(imgcount):
    img  = np.zeros((height,width),np.uint8)

    f.read(1) # read gap byte

    data = array.array('h')

    ###  buf = f.read(imgsize*2)
    ###  data.fromstring(buf)

    data.fromfile(f,imgsize)

    for r in range(height):
      for c in range(width):
        img[r,c] = data[r * width + c]

    img = cv2.resize(img, (0,0), fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    rand_x = random.randint(0, (32-28))
    rand_y = random.randint(0, (32-28))
    clopped_img = img[rand_x:rand_x+28, rand_y:rand_y+28]
    filename = "./data/" + args.folder + "/" + fn.split("/")[-1].replace(".vec","") + "_" + str(i) + ".png"
    cv2.imwrite(filename, clopped_img)
    
showvec(args.img)