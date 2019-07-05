import numpy as np
import matplotlib
import pickle
import datetime

from keras.layers import Conv2D, Dense, Input,  MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
from keras import regularizers
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
num_epochs = 50
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

inputLayer = Input(shape=(inputLayer_size[0], inputLayer_size[1], inputLayer_size[2]))  # adapt this if using `channels_first` image data format

autoencoderModel = Sequential()
#model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
#model.add(MaxPooling2D(pool_size=(2, 2)))

# Build the model ass Sequential()
autoencoderModel.add(Conv2D(16, (3, 3), activation=hiddenLayer_activationFunction, input_shape=inputLayer_size, padding=padding))
autoencoderModel.add(MaxPooling2D((2, 2), padding=padding))
autoencoderModel.add(Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding))
autoencoderModel.add(MaxPooling2D((2, 2), padding=padding))
autoencoderModel.add(Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding))
autoencoderModel.add(MaxPooling2D((2, 2), padding=padding)) # after this we are at the Encoder level

# Decoder
autoencoderModel.add(Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding))  # or after this???
autoencoderModel.add(UpSampling2D((2, 2)))
autoencoderModel.add(Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding))
autoencoderModel.add(UpSampling2D((2, 2)))
autoencoderModel.add(Conv2D(16, (3, 3), activation=hiddenLayer_activationFunction))
autoencoderModel.add(UpSampling2D((2, 2)))
autoencoderModel.add(Conv2D(1, (3, 3), activation=outputLayer_activationFunction, padding=padding))

# x = Conv2D(16, (3, 3), activation=hiddenLayer_activationFunction, padding=padding)(inputLayer)
# x = MaxPooling2D((2, 2), padding=padding)(x)
# x = Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding)(x)
# x = MaxPooling2D((2, 2), padding=padding)(x)
# x = Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding)(x)
# encodedLayer = MaxPooling2D((2, 2), padding=padding)(x)
#
# # at this point the representation is (4, 4, 8) i.e. 128-dimensional
#
# x = Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding)(encodedLayer)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding)(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(16, (3, 3), activation=hiddenLayer_activationFunction)(x)
# x = UpSampling2D((2, 2))(x)
# outputLayer = Conv2D(1, (3, 3), activation=outputLayer_activationFunction, padding=padding)(x)
#
#
# # Create Model
# autoencoderModel = Model(inputLayer, outputLayer)
# encoderModel = Model(inputLayer, encodedLayer)
# #tmpDecoderLayer = autoencoderModel.layers[-1]
# #decoderModel = Model(encodedLayer, tmpDecoderLayer(encodedLayer))
# autoencoderModel.compile(optimizer=optimizer, loss=loss)

autoencoderModel.summary()
# print("Now the Sequential model summary")
# autoencoderModel.summary()

# Train Model
# we train it
autoencoderModel.compile(optimizer=optimizer, loss=loss)
autoencoderModel.fit(x_train, x_train, epochs=num_epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test,x_test), callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# Copy only the encoder part into a seperate model
weights0 = autoencoderModel.layers[0].get_weights()
weights1 = autoencoderModel.layers[1].get_weights()
weights2 = autoencoderModel.layers[2].get_weights()
weights3 = autoencoderModel.layers[3].get_weights()
weights4 = autoencoderModel.layers[4].get_weights()
weights5 = autoencoderModel.layers[5].get_weights()
weights6 = autoencoderModel.layers[6].get_weights()

encoderOnlyModel = Sequential()
encoderOnlyModel.add(Conv2D(16, (3, 3), activation=hiddenLayer_activationFunction, input_shape=inputLayer_size, padding=padding, weights=weights0))
encoderOnlyModel.add(MaxPooling2D((2, 2), padding=padding, weights=weights1))
encoderOnlyModel.add(Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding, weights=weights2))
encoderOnlyModel.add(MaxPooling2D((2, 2), padding=padding,weights=weights3))
encoderOnlyModel.add(Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding, weights=weights4))
encoderOnlyModel.add(MaxPooling2D((2, 2), padding=padding, weights=weights5))
encoderOnlyModel.add(Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding, weights=weights6))

encoderOnlyModel.summary()

# model2 = Sequential()
# model2.add(Dense(20, 64, weights=model.layers[0].get_weights()))
# model2.add(Activation('tanh'))
#
# autoencoderModel.add(Conv2D(16, (3, 3), activation=hiddenLayer_activationFunction, input_shape=inputLayer_size, padding=padding))
# autoencoderModel.add(MaxPooling2D((2, 2), padding=padding))
# autoencoderModel.add(Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding))
# autoencoderModel.add(MaxPooling2D((2, 2), padding=padding))
# autoencoderModel.add(Conv2D(8, (3, 3), activation=hiddenLayer_activationFunction, padding=padding))
# autoencoderModel.add(MaxPooling2D((2, 2), padding=padding)) # after this we are at the Encoder level


# Retreive test Images (pre,post)
decoded_imgs = autoencoderModel.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].squeeze())
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].squeeze())
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


if saveModel == 1:
    with open(filename, "w") as f:  # The w stands for write
        pickle.dump([autoencoderModel, encoderOnlyModel], f)
    f.close()

abc = 1

