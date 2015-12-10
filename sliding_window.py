import numpy as np

def slide_window(img, wsize, stride, block):
	point_r = 0
	r_size, c_size = img.shape
	
	while point_r+wsize < r_size:
		point_c = 0
		while point_c+wsize < c_size:
			block(img[point_r:point_r+wsize, point_c:point_c+wsize], point_r, point_c, wsize)
			point_c += stride
		point_r += stride

# def print_array(array):
# 	print array
# 	print "------------------------------"

# test_arr = np.array(range(100)).reshape(10,10)

# slide_window(test_arr, 3, 2, print_array)
