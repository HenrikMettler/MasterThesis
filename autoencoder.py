import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras import regularizers
from keras.datasets import fashion_mnist
#from old_AutoencoderClass import Autoencoder
import matplotlib
matplotlib.use("TkAgg") # matplotlib import adapted for use on MacOS
from matplotlib import pyplot as plt

# Import data
(x_train, _), (x_test, _) = fashion_mnist.load_data()

# Prepare Input: Normalizing, Flatten Images
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


#  Network Parameters, Train Parameters
inputLayer_size = x_train.shape[1]
hiddenLayer_size = 32 # default from keras documentation
hiddenLayer_activationFunction = 'relu' # default from keras documentation
outputLayer_activationFunction = 'sigmoid' # default from keras documentation
outputLayer_size = inputLayer_size
optimizer ='adadelta' # default from keras documentation
loss ='binary_crossentropy' # default from keras documentation
batch_size = 256 # default from keras documentation
num_epochs = 50  # default from keras documentation

# Regularization Parameter
regularizationParameter = regularizers.l1(10e-5)

# Create Dependant Parameters
inputLayer = Input(shape=(inputLayer_size,))
hiddenInput = Input(shape=(hiddenLayer_size,))
hiddenLayer = Dense(hiddenLayer_size, activation=hiddenLayer_activationFunction,activity_regularizer=regularizationParameter)(inputLayer)
outputLayer = Dense(outputLayer_size, activation=outputLayer_activationFunction)(hiddenLayer)

# Create Model
autoencoderModel = Model(inputLayer,outputLayer)
encoderModel = Model(inputLayer, hiddenLayer)
tmpDecoderLayer = autoencoderModel.layers[-1]
decoderModel = Model(hiddenInput, tmpDecoderLayer(hiddenInput))
autoencoderModel.compile(optimizer=optimizer, loss=loss)

# Train Model
autoencoderModel.fit(x_train,x_train,epochs=num_epochs,batch_size=batch_size,shuffle=True,validation_data=(x_test,x_test))

# Retreive test Images (pre,post)
encodedImages= encoderModel.predict(x_test)
decodedImages = decoderModel.predict(encodedImages)

# plot results
plt.figure(figsize=(20, 4))
for i in range(10):
    # Original
    subplot = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    subplot.get_xaxis().set_visible(False)
    subplot.get_yaxis().set_visible(False)

    # Reconstruction
    subplot = plt.subplot(2, 10, i + 11)
    plt.imshow(decodedImages[i].reshape(28, 28))
    plt.gray()
    subplot.get_xaxis().set_visible(False)
    subplot.get_yaxis().set_visible(False)
plt.show()