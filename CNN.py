
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from skimage import filters

# def preprocess(x,y):
# 	x = tf.cast(x,tf.float32)
# 	y = tf.cast(y,tf.int64)
# 	return x,y

# def create_dataset(xs, ys, n_classes):
# 	ys = tf.one_hot(ys,depth=n_classes)
# 	return tf.data.Dataset.from_tensor_slices((xs, ys)) \
# 		# .map(preprocess) \
# 		# .shuffle(len(ys)) \
# 		.batch(128)



# choose the data to use; there are 15000 images, but the samples only have
# 150 images; preprocessed means the images are centered and rescaled
# directory = 'data/original_sample'
# directory = 'data/preprocessed_sample'
# directory = 'data/original'
directory = 'data/preprocessed'

# preallocate feature matrix X and label vector d
num_pixels_x = 64
num_pixels_y = 64
num_images = 15000
X = np.zeros([num_images,num_pixels_x,num_pixels_y])
d = np.zeros([num_images,])

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

	X[i,:,:] = img
	d[i] = label



from sklearn.model_selection import train_test_split

# split data into training and testing data
X_train, X_test, d_train, d_test = train_test_split(X,d,test_size=0.2) # train size is 12000



import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

# reshape data to have a single channel/color
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
# determine the shape of the input images
in_shape = X_train.shape[1:]
print(in_shape)

# preprocess the data a bit
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
d_train -= 1 # to go from [1,15] to [0,14]
d_test -= 1

# define model
model = Sequential()
model.add(Conv2D(60,(6,6),activation='relu',kernel_initializer='he_uniform',input_shape=(64,64,1)))
model.add(MaxPool2D((3,3)))
model.add(Flatten())
model.add(Dense(200, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.3))
model.add(Dense(15, activation='softmax'))

# model.add(Reshape(target_shape=(64*64,),input_shape=(64,64)))
# model.add(Dense(units=200,activation='relu'))
# model.add(Dense(units=15,activation='softmax'))



# define loss and optimizer
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# fit the model
model.fit(X_train,d_train,epochs=10,batch_size=128,verbose=0)

# evaluate the model
loss,acc = model.evaluate(X_test,d_test,verbose=0)

err = 100*(1-acc)
print(np.round(err,2))
