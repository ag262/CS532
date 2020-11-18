
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from skimage import filters
from skimage.measure import regionprops
from skimage import transform
from skimage.util import img_as_ubyte
from skimage import exposure

# preprocess all image files
directory = 'data/original'
for f_name in os.listdir(directory):

	# load image
	img = io.imread(directory + '/' + f_name)

	# io.imshow(img)
	# io.show()

	# do otsu filtering
	threshold_value = filters.threshold_otsu(img)
	labeled_foreground = (img > threshold_value).astype(int)
	properties = regionprops(labeled_foreground, img)

	# io.imshow(img,cmap='gray')
	# io.show()

	# get the bounding box coordinates (i.e. indices) for rows and cols
	min_r, min_c, max_r, max_c = properties[0].bbox
	H = max_r-min_r
	W = max_c-min_c
	max_dim = max(H,W)
	center_r = min_r + int(H/2)
	center_c = min_c + int(W/2)
	radius = int(max_dim*1.2/2)

	# add vertical columns of zeros to the left and right
	if H > W:
		min_r_new = max(center_r-radius,0)
		max_r_new = min(center_r+radius,labeled_foreground.shape[0])
		img_cropped = labeled_foreground[min_r_new:max_r_new,min_c:max_c]
		H = max_r_new-min_r_new
		W = max_c-min_c
		num_to_be_added = H-W
		num_left = int(num_to_be_added/2)
		num_right = num_to_be_added-num_left
		img_cropped = np.hstack((np.zeros([H,num_left]),img_cropped,np.zeros([H,num_right])))
	# or add horizontal rows of zeros to the top and bottom
	elif W > H:
		min_c_new = max(center_c-radius,0)
		max_c_new = min(center_c+radius,labeled_foreground.shape[1])
		img_cropped = labeled_foreground[min_r:max_r,min_c_new:max_c_new]
		H = max_r-min_r
		W = max_c_new-min_c_new
		num_to_be_added = W-H
		num_top = int(num_to_be_added/2)
		num_bottom = num_to_be_added-num_top
		img_cropped = np.vstack((np.zeros([num_top,W]),img_cropped,np.zeros([num_bottom,W])))
	else:
		min_r_new = max(center_r-radius,0)
		max_r_new = min(center_r+radius,labeled_foreground.shape[0])
		min_c_new = max(center_c-radius,0)
		max_c_new = min(center_c+radius,labeled_foreground.shape[1])
		img_cropped = labeled_foreground[min_r_new:max_r_new,min_c_new:max_c_new]

	# resize the img (redo the otsu filtering only when the file is opened again)
	img = transform.resize(img_cropped, (64, 64))
	img = exposure.rescale_intensity(img)

	# io.imshow(img,cmap='gray')
	# io.show()

	# save the preprocessed image with the same name but in a different folder
	img = io.imsave('data/preprocessed/' + f_name,img_as_ubyte(img))
