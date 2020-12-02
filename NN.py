
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from skimage import filters

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import plot_confusion_matrix
# from mlxtend.plotting import plot_decision_regions

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU



# choose the data to use; there are 15000 images, but the samples only have
# 150 images; preprocessed means the images are centered and rescaled
# directory = 'data/original_sample'
# directory = 'data/preprocessed_sample'
# directory = 'data/original'
directory = 'data/preprocessed'

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



# split data into nontesting and testing data
X_nontest, X_test, d_nontest, d_test = train_test_split(X,d,test_size=0.2) # train size is 12000

batch_size = 16
epochs = 10
num_classes = 15
