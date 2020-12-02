
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from skimage import filters

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import plot_confusion_matrix
# from mlxtend.plotting import plot_decision_regions



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

# hyperparameter tuning

lam_arr = np.array([1.0,5.0,10.0])
CV_err_arr = np.array([])
CV_var_arr = np.array([])

for lam in lam_arr:

	# 10-fold cross validation

	num_split = 10
	skf = StratifiedKFold(n_splits=num_split)
	err_arr = np.array([])

	for train_ind,valid_ind in skf.split(X_nontest,d_nontest):

		X_train = X_nontest[train_ind]
		d_train = d_nontest[train_ind]
		X_valid = X_nontest[valid_ind]
		d_valid = d_nontest[valid_ind]

		# Create the kernel SVM
		svm = SVC(C=lam,kernel='rbf',tol=1e-3,decision_function_shape='ovr')

		# Make it an OvR ovr_classifier
		ovr_classifier = OneVsRestClassifier(svm)

		# Fit the data to the OvR classifier
		ovr_classifier = ovr_classifier.fit(X_train,d_train)

		err = 100*( 1-ovr_classifier.score(X_test,d_test.ravel()) )

		err_arr = np.append(err_arr,err)

	CV_err_arr = np.append(CV_err_arr,np.mean(err_arr))
	CV_var_arr = np.append(CV_var_arr,np.var(err_arr))

print(np.round(CV_err_arr,2))
print(np.round(np.sqrt(CV_var_arr),2))
