#!/usr/bin/python
import glob
import hashlib
from time import sleep
import numpy as np
import os.path as path
import scipy
import scipy.signal
from sys import exit,argv,stdout
from optparse import OptionParser
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

from skimage import io,color
from skimage import exposure

"""
Some experimentation with kera's/tensorflows operations to detail what they do
"""
def sharpen(image_path):
	convolve2d(image_path,kernel=np.array([[0,1,0],[1,-5,1],[0,1,0]]))

def edge_detect3(image_path):  #this is my favourite one!
	convolve2d(image_path,kernel=np.array([[1,0,-1],[0,0,0],[-1,0,1]]))

def edge_detect2(image_path):
	convolve2d(image_path,kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]]))

def edge_detect(image_path):
	convolve2d(image_path,kernel=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]))

def blur_denoise(image_path):
	convolve2d(image_path,kernel=np.array([[1,1,1],[1,1,1],[1,1,1]])/9.0)

def convolve2d(image_path,kernel=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])):
	img = io.imread(image_path)	
	img = color.rgb2gray(img)

	edges = scipy.signal.convolve2d(img,kernel,'valid') #valid means 0 padding

	#show result
	fig = plt.figure()

	ax1 = fig.add_subplot(1,2,1)
	ax1.imshow(edges)
	plt.title("conv")
	
		
	ax2 = fig.add_subplot(1,2,2)
	ax2.imshow(img)
	plt.title("base")

	plt.show()
	
class ExperimentalCNNModel_1:
	def __init__(self,n_nodes=32,\
						n_layers=1, \
						kernel=(3,3), \
						activation='relu', \
						images_shape=(None,20,20)):
		self.n_layers = n_layers
		self.max_nodes = 100
		self.kernel_size = kernel
		self.activation_function = activation
		self.images_shape = images_shape
		self.n_nodes = n_nodes 
		self.model = Sequential()
		self.loss="binary_crossentropy"
		self.optimizer = "adam"
		self.monitor_metrics = ['accuracy']

	def build(self):
		#build network
		self.model.add(Conv2D(self.n_nodes,\
										self.kernel_size,\
										activation=self.activation_function))
		self.model.add(Dense(1))
		self.network_build = True
	def compile(self):
			if self.network_build:
				self.model.compile(loss=self.loss,\
										optimizer=self.optimizer,\
										metrics=self.monitor_metrics) 
				self.compiled = True
				return self.model
			else:
				raise Exception("model not build before compile")

	def summary(self):
		if self.model and self.network_build and self.compiled:
			self.model.summary()
		return
if __name__=="__main__":
	parser = OptionParser()
	parser.add_option("-e",\
							"--extension",\
							help="file extension",\
							type="string",\
							dest="file_extension")	
	parser.add_option("-p",\
							"--path",\
							help="path to image files",\
							type="string",\
							dest="file_path")
	"""
	parser.add_option("-n",\
							"--image_limit",\
							help="number of images to process",\
							type="integer",\
							dest="image_limit")
	"""

	options, arguments = parser.parse_args()
	file_extension = options.file_extension
	file_path = options.file_path

	print("[*] Starting data processing ...")
	file_paths = glob.glob(path.join(file_path,\
												"*.%s" % (file_extension)))
	for file_path in file_paths:
		#blur_denoise(file_path)

		edge_detect(file_path)
		edge_detect2(file_path)
		edge_detect3(file_path)
