#!/usr/bin/env python
# coding: utf-8

# Imports
# get_ipython().run_line_magic('matplotlib', 'inline')

# Import matplotlib
import matplotlib

# Force matplotlib to not use any Xwindows backend
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import globals
import itertools
import numpy as np

# Call function confusionMatrix
def confusionMatrix(predict,
                    descriptor_name,
					dataset_name,
                    model_name):
	# SVM Model

	# Define Labels and Rotation of plt.xticks
	if dataset_name == 'CIFAR-10':
		# Labels
		labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

		# Rotation
		rotation = 75
		
	elif (dataset_name == 'Extended-CK+') or (dataset_name == 'FER-2013') or (dataset_name == 'JAFFE'):
		# Labels
		labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

		# Rotation
		rotation = 75

	elif dataset_name == 'FEI':
		# Labels
		labels = ['happy', 'neutral']

		# Rotation
		rotation = 75

	elif dataset_name == 'MNIST':
		# Labels
		labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

		# Rotation
		rotation = 0

	# Turn interactive plotting off
	plt.ioff()

	# Create a new figure
	plt.figure()

	# Get the Confusion Matrix using Sklearn
	cm = confusion_matrix(y_true = globals.test_feature_vec[1],
						  y_pred = predict,
						  labels = labels)

	# Print the Confusion Matrix as text
	# print(cm)

	# Plot the Confusion Matrix as an image
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j,
				i,
				cm[i, j],
				horizontalalignment = 'center',
				color = 'white' if cm[i, j] > thresh else 'black')

	# Make various adjustments to the plot
	plt.imshow(cm, interpolation='nearest', cmap='Greys')
	plt.colorbar()
	tick_marks = np.arange(globals.num_classes)
	plt.xticks(tick_marks, labels, rotation = rotation)
	plt.yticks(tick_marks, labels)
	plt.title('Confusion Matrix')
	plt.xlabel('Predicted Label')
	plt.ylabel('True Label')

	# Save it
	plt.savefig('Figures/Datasets/%s/%s-%s-confusion-matrix.png' % (dataset_name, descriptor_name, model_name),
				bbox_inches = 'tight',
				transparent = True,
				dpi = 300)

	# Close it
	plt.close()