import numpy as np
import matplotlib
import pickle
import datetime

from keras.layers import Conv2D, Dense, Input,  MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import TensorBoard
from functions import *

matplotlib.use("TkAgg") # matplotlib import adapted for use on MacOS

from matplotlib import pyplot as plt

# parameters
testFraction = 0.1
# boolean for model saving
saveModel = 1
# generate filename and location for storage
currentTime = datetime.datetime.now()
helperStr = ("autoencoder_objects%s" % currentTime)
filename = (helperStr + ".pickle")


# create a list of images from data set
imageList, imageList_asArray = create_imageList()


# clean image list from black and white images
imageList_asArray = eliminate_nonRGB(imageList_asArray)

# zero pad images to square size
imageList_asArray = zeroPad2Square(imageList_asArray)

# up-/down-sample image size to mean size
imageList_asArray = rescale_imageList(imageList_asArray)

# split data in train and test with desired percentages
trainSet, testSet = create_trainSet_testSet(imageList_asArray, testFraction)

# data normalization (rescaling from [0-255] to [0-1]
trainSet, testSet = normalize_data(trainSet, testSet) # ToDo: check why the division by float makes everything so slow!

