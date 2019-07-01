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

matplotlib.use("TkAgg") # matplotlib import adapted for use on MacOS

from matplotlib import pyplot as plt


# generate filename and location for storage
currentTime = datetime.datetime.now()
helperStr = ("autoencoder_mnist%s" % currentTime)
filename = (helperStr + ".pickle")

# boolean for model saving
saveModel = 1
# Import data
(x_train, _), (x_test, _) = mnist.load_data()

# Prepare Input: Normalizing, Flatten Images
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
""" For autoencoder w/o convolution
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) 
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))"""
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format


#  Network Parameters, Train Parameters
inputLayer_size = np.array([x_train.shape[1],x_train.shape[2],x_train.shape[3]])
#hiddenLayer_size = 32 # default from keras documentation
hiddenLayer_activationFunction = 'relu' # default from keras documentation
outputLayer_activationFunction = 'sigmoid' # default from keras documentation
outputLayer_size = inputLayer_size
optimizer ='adadelta' # default from keras documentation
loss ='binary_crossentropy' # default from keras documentation
batch_size = 256
num_epochs = 20
padding = 'same' # default from keras documentation

# Regularization Parameter
regularizationParameter = regularizers.l1(10e-5)

''' In comments: first Autoencoder without convolutional layers 
# Create Dependant Parameters
inputLayer = Input(shape=(inputLayer_size,))
hiddenInput = Input(shape=(hiddenLayer_size,))
hiddenLayer = Dense(hiddenLayer_size, activation=hiddenLayer_activationFunction,activity_regularizer=regularizationParameter)(inputLayer)
outputLayer = Dense(outputLayer_size, activation=outputLayer_activationFunction)(hiddenLayer)
'''

# Create convolutional downsampling with two layers
# Todo: Adapt two Colored images (more than 1D in 3rd column of image array)
inputLayer = Input(shape=(inputLayer_size[0], inputLayer_size[1], inputLayer_size[2]))  # adapt this if using `channels_first` image data format
# Todo: Adapt layer size to diff image sizes
x = Conv2D(16, (3, 3), activation=hiddenLayer_activationFunction, padding=padding)(inputLayer)
x = MaxPooling2D((2, 2), padding=padding)(x)
x = Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding)(x)
x = MaxPooling2D((2, 2), padding=padding)(x)
x = Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding)(x)
encodedLayer = MaxPooling2D((2, 2), padding=padding)(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding)(encodedLayer)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding)(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation=hiddenLayer_activationFunction)(x)
x = UpSampling2D((2, 2))(x)
outputLayer = Conv2D(1, (3, 3), activation=outputLayer_activationFunction, padding=padding)(x) # Todo: check what size the output now takes


# Create Model
autoencoderModel = Model(inputLayer, outputLayer)
encoderModel = Model(inputLayer, encodedLayer)
#tmpDecoderLayer = autoencoderModel.layers[-1]
#decoderModel = Model(encodedLayer, tmpDecoderLayer(encodedLayer))
autoencoderModel.compile(optimizer=optimizer, loss=loss)

autoencoderModel.summary()

# Train Model
autoencoderModel.fit(x_train, x_train, epochs=num_epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test,x_test), callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# Retreive test Images (pre,post)
decoded_imgs = autoencoderModel.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i])
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


if saveModel == 1:
    with open(filename, "w") as f:  # The w stands for write and the b stands for binary mode (used for non-text files)
        pickle.dump(autoencoderModel, f)

abc = 1

