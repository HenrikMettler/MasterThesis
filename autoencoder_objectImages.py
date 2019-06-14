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
from importObjectData import *

matplotlib.use("TkAgg") # matplotlib import adapted for use on MacOS

from matplotlib import pyplot as plt

imageList = create_imageList()





a = 4