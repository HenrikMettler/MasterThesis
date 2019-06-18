import numpy as np
import matplotlib
import pickle
import datetime

from keras.layers import Conv2D, Dense, Input,  MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers
#from keras.datasets import fashion_mnist
#from keras.datasets import mnist
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
imageList, mean_imageSize = preprocess_imageList()

# split data in train and test with desired percentages
trainSet, testSet = create_trainSet_testSet(imageList, testFraction)

# data normalization (rescaling from [0-255] to [0-1]
trainSet, testSet = normalize_data(trainSet, testSet) # ToDo: check why the division by float makes everything so slow!

# reshape trainSet and testSet to 4d arrays (numSamples, 300,300,3)
x_train = np.zeros((len(trainSet), mean_imageSize,mean_imageSize,3))
x_test = np.zeros((len(testSet), mean_imageSize,mean_imageSize,3))
#  this is very ugly for know, but works... todo: make nice
for idxtrainSet in range(len(trainSet)):
    currentImage = trainSet[idxtrainSet]
    for idxHeight in range(300):
        currentRow = currentImage[idxHeight]
        x_train[idxtrainSet,idxHeight,:,:] = currentRow

for idxtestSet in range(len(testSet)):
    currentImage = testSet[idxtestSet]
    for idxHeight in range(300):
        currentRow = currentImage[idxHeight]
        x_test[idxtestSet, idxHeight, :, :] = currentRow

#  Network Parameters, Train Parameters
inputLayer_size = mean_imageSize
#hiddenLayer_size = 32 # default from keras documentation
hiddenLayer_activationFunction = 'relu' # default from keras documentation
outputLayer_activationFunction = 'sigmoid' # default from keras documentation
outputLayer_size = inputLayer_size
optimizer ='adadelta' # default from keras documentation
loss ='binary_crossentropy' # default from keras documentation
batch_size = 256 # default from keras documentation
num_epochs = 500
padding = 'same' # default from keras documentation

# Regularization Parameter
regularizationParameter = regularizers.l1(10e-5)


# Create convolutional downsampling with two layers
inputLayer = Input(shape=(inputLayer_size, inputLayer_size, 3))  # adapt this if using `channels_first` image data format

# Todo: Adapt two Colored images (more than 1D in 3rd column of image array)
x1 = Conv2D(512, (4, 4), activation=hiddenLayer_activationFunction, padding=padding)(inputLayer)
x2 = MaxPooling2D((4, 4), padding=padding)(x1)
x3 = Conv2D(64, (4, 4), activation=hiddenLayer_activationFunction, padding=padding)(x2)
x4 = MaxPooling2D((4, 4), padding=padding)(x3)
x5 = Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding)(x4)
encodedLayer = MaxPooling2D((2, 2), padding=padding)(x5)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x6 = Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding)(encodedLayer)
x7 = UpSampling2D((2, 2))(x6)
x8 = Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding)(x7)
x9 = UpSampling2D((4, 4))(x8)
x10 = Conv2D(64, (4, 4), activation=hiddenLayer_activationFunction)(x9)
x11 = UpSampling2D((4, 4))(x10)
outputLayer = Conv2D(1, (3, 3), activation=outputLayer_activationFunction, padding=padding)(x11) # Todo: check what size the output now takes

# Create Model
autoencoderModel = Model(inputLayer, outputLayer)
encoderModel = Model(inputLayer, encodedLayer)
#tmpDecoderLayer = autoencoderModel.layers[-1]
#decoderModel = Model(encodedLayer, tmpDecoderLayer(encodedLayer))
autoencoderModel.compile(optimizer=optimizer, loss=loss)

# Train Model
autoencoderModel.fit(x_train, x_train, epochs=num_epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test,x_test), callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# Retreive test Images (pre,post)
decoded_imgs = autoencoderModel.predict(testSet)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(testSet[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
