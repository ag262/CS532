
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from skimage import filters



# choose the data to use; there are 15000 images, but the samples only have
# 150 images; preprocessed means the images are centered and rescaled
# directory = 'data/original_sample'
# directory = 'data/preprocessed_sample'
directory = 'data/original'
# directory = 'data/preprocessed'

# preallocate feature matrix X and label vector d
num_pixels = 64*64
num_images = 15000
X = np.zeros([num_images,num_pixels])
d = np.zeros([num_images,1])

# fill in X and d
for (i, f_name) in enumerate(os.listdir(directory)):

	# load image
	img = io.imread(directory + '/' + f_name)

	# io.imshow(img,cmap='gray')
	# io.show()

	# apply otsu filtering
	threshold_value = filters.threshold_otsu(img)
	img = (img > threshold_value).astype(int)

	# io.imshow(img,cmap='gray')
	# io.show()

	# labels can only have 1 or 2 digits
	if f_name[-6] != '_':
		label = int(f_name[-6] + f_name[-5])
	else:
		label = int(f_name[-5])

	X[i,:] = img.flatten()
	d[i] = label


temp = np.hstack((X,d))
np.random.shuffle(temp)
np.random.shuffle(temp)
X = temp[:,:-1]
d = temp[:,-1]

# 1 for "train", 2 for test
num_train = 10000
# X1 = X[:num_train,:]
# X2 = X[num_train:,:]
# d1 = d[:num_train]
# d2 = d[num_train:]
k = 500

num_err = 0
for i in range(num_train,num_images):

	# get indices of the k NN
	dist_to_neighbors = np.linalg.norm(X[:num_train,:]-X[i,:],axis=1)
	inds = np.argsort(dist_to_neighbors)[:k]

	# use the number of occurrences of codes of each class to choose the code of
	# the ith label (codes go from 1 through 15)
	classes = d[inds].astype(int)
	occurrences = np.bincount(classes)
	code = np.argmax(occurrences)

	if code != d[i]:
		num_err += 1

err_rate = num_err/(num_images-num_train)
print('Error Rate = ' + str(round(err_rate,3)))
