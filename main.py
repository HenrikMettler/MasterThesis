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

filename = "autoencoder_mnist2019-07-01 14:30:09.611826.pickle"

infile = open(filename,'rb')
loadedAutoencoder = pickle.load(infile)
infile.close()

