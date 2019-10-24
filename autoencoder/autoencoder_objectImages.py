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

# define the path to the dataset, containing the different image categories as subfolders
ground_data_path = "/Users/henrik.mettler/Desktop/DataSetForAutoencoder"

# parameters
testFraction = 0.1
# boolean for model saving
saveModel = 1
# generate filename and location for storage
currentTime = datetime.datetime.now()
helperStr = ("autoencoder_objects%s" % currentTime)
filename = (helperStr + ".pickle")


# create a list of images from data set
imageList, categoryList, mean_imageSize = preprocess_imageList(ground_data_path)

# split data in train and test with desired percentages
trainSet, testSet = create_trainSet_testSet(imageList, testFraction)

# data normalization (rescaling from [0-255] to [0-1]
trainSet, testSet = normalize_data(trainSet, testSet) #

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
num_conv_layer = 3 # for now the architecture below is fixed for num_conv_layer = 3
num_pool_layer = num_conv_layer
filter_sizes = [4, 8, 16]
kernel_sizes = [12, 8, 4]
maxpool_sizes = [8, 4, 2]
hiddenLayer_activationFunction = 'relu' # same across all layers
outputLayer_activationFunction = 'sigmoid' # default from keras documentation
outputLayer_size = inputLayer_size # redundant info
optimizer ='adadelta' # default from keras documentation
loss ='binary_crossentropy' # default from keras documentation
batch_size = 4
num_epochs = 25
padding = 'same' # default from keras documentation

# Regularization Parameter
regularizationParameter = regularizers.l1(10e-5)


# Create convolutional downsampling with two layers
inputLayer = Input(shape=(inputLayer_size, inputLayer_size, 3))  # adapt this if using `channels_first` image data format

x1 = Conv2D(filters=filter_sizes[0], kernel_size= kernel_sizes[0] , activation=hiddenLayer_activationFunction, padding='same', data_format="channels_last")(inputLayer)
x2 = MaxPooling2D(maxpool_sizes[0], padding=padding)(x1)
x3 = Conv2D(filters=filter_sizes[1], kernel_size= kernel_sizes[1] , activation=hiddenLayer_activationFunction, padding='same', data_format="channels_last")(x2)
x4 = MaxPooling2D(maxpool_sizes[1], padding=padding)(x3)
x5 = Conv2D(filters=filter_sizes[2], kernel_size= kernel_sizes[2] , activation=hiddenLayer_activationFunction, padding='same', data_format="channels_last")(x4)
encodedLayer= MaxPooling2D(maxpool_sizes[2], padding=padding)(x5)

# x7 = Conv2D(filters=filter_sizes[3], kernel_size= kernel_sizes[3] , activation=hiddenLayer_activationFunction, padding='same', data_format="channels_last")(x6)
# encodedLayer = MaxPooling2D(maxpool_sizes[3], padding=padding)(x7)

# at this point the representation is ~ input/(prod(maxpool))^2 *filter[end] eg: 300^2/(2*2*2)^2*8=-dimensional

# x8 = Conv2D(filters=filter_sizes[3], kernel_size= kernel_sizes[3], activation=hiddenLayer_activationFunction, padding=padding)(encodedLayer)
# x9 = UpSampling2D(maxpool_sizes[3])(x8)
x10 = Conv2D(filters=filter_sizes[2], kernel_size= kernel_sizes[2], activation=hiddenLayer_activationFunction, padding= padding)(encodedLayer)
x11= UpSampling2D(maxpool_sizes[2])(x10)
x12 = Conv2D(filters=filter_sizes[1], kernel_size= kernel_sizes[1], activation=hiddenLayer_activationFunction,padding= padding)(x11)
x13 = UpSampling2D(maxpool_sizes[1])(x12)
x14 = Conv2D(filters=filter_sizes[0], kernel_size= kernel_sizes[0], activation=hiddenLayer_activationFunction,padding= padding)(x13)
x15 = UpSampling2D(maxpool_sizes[0])(x14)
outputLayer = Conv2D(3, (3, 3), activation=outputLayer_activationFunction, padding=padding)(x15)


# Create Model
autoencoderModel = Model(inputLayer, outputLayer)
encoderModel = Model(inputLayer, encodedLayer)
#tmpDecoderLayer = autoencoderModel.layers[-1]
#decoderModel = Model(encodedLayer, tmpDecoderLayer(encodedLayer))
autoencoderModel.compile(optimizer=optimizer, loss=loss)

# print a summary of the autoencoder model to the console
print('Summary of the Autoencoder model with layer shapes')
autoencoderModel.summary()

dimOutput = 320# Todo: change to get dim info from output layer as int
# adapt train and test data for output layer to output layer size (zero padding at the edges ToDo: alter hardcoded struct
x_train_output = zeroPad2OutputSize(x_train, dimOutput)
x_test_output = zeroPad2OutputSize(x_test, dimOutput)

# Train Model
autoencoderModel.fit(x_train, x_train_output, epochs=num_epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, x_test_output), callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# Retreive test Images (pre,post)
decoded_imgs = autoencoderModel.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_output[7*i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[7*i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

a = 7