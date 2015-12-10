import cv
import cv2
import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils
import chainer.functions as F
import six
import argparse
import sliding_window as sw
from PIL import Image
from PIL import ImageOps

parser = argparse.ArgumentParser(
    description='A Neural Algorithm of Artistic Style')
parser.add_argument('--img', '-i', default='',
                    help='path of input image')
args = parser.parse_args()

with open('trained_model.pkl', 'rb') as model_pickle:
  model = six.moves.cPickle.load(model_pickle)

ratio = 0.7

def forward(x_data):
  x = Variable(x_data)
  h1 = F.relu(model.l1(x))
  h2 = F.relu(model.l2(h1))
  h3 = F.relu(model.l3(h2))
  y = F.softmax(model.l4(h3))
  return y

img = Image.open(args.img)
(iw, ih) = img.size
print iw
print ih

new_width = int(iw*ratio) 
new_height = int(ih*ratio)
print new_width
print new_height

original_img = cv2.imread(args.img)

original_img = cv2.resize(original_img, (new_width, new_height))

img = cv2.cvtColor(original_img, cv2.cv.CV_BGR2GRAY)
img = np.asarray(img).astype(np.float32) / 255.

def judge(array, r, c, wsize):

  xd = np.asarray(array).reshape((1,784)).astype(np.float32)
  yd = forward(xd).data[0]
  prob = yd[1]/(yd[0]+yd[1])
  threshold = 0.8
  if(prob > threshold):
    print "------ detected (" + str(c) + "," + str(r) + ")  P(" + str(prob) + " )-------------"
    cv2.rectangle(original_img, (c, r), (c+wsize, r+wsize), (0,255,0), 3) 

sw.slide_window(img, 28, 14, judge)

cv2.imwrite('./result.png',original_img)
cv2.imshow('img',original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

