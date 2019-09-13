import os
import numpy as np
# import pickle
from keras.layers import LSTM, LSTMCell, Dense
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import RMSprop

#from PIL import Image
from create_imageObject import *


def create_imageList(ground_data_path):
    # path to all data
    ground_data_path = "/Users/henrik.mettler/Desktop/AutoencoderDataSet" # Adapt this to local path
    # folder paths
    path_list = os.listdir(ground_data_path)

    imageList = []
    categoryList = []
    if '.DS_Store' in path_list:
        path_list.remove('.DS_Store') # avoid having a unusable folder

    for idxFolder in range(len(path_list)):
        currentFolderName = path_list[idxFolder]
        currentFolderPath = ground_data_path + '/' + currentFolderName
        fileList = os.listdir(currentFolderPath)
        if '.DS_Store' in fileList:
            fileList.remove('.DS_Store') # avoid having a unusable file
        numberFiles = len(fileList)
        label = idxFolder + 1

        for idxFile in range(numberFiles):
            currentFileName = currentFolderPath + '/' + fileList[idxFile]
            currentImage = Image.open(currentFileName,'r')

            currentImage_asArray = np.asarray(currentImage)
            imageList.append(currentImage_asArray)
            categoryList.append(currentFolderName)
            a = label

    return imageList, categoryList


def create_trainSet_testSet(imageList, testFraction):

    trainSet = []
    testSet = []
    for idxImage in range(len(imageList)):
        p = np.random.random()
        if p >= testFraction:
            trainSet.append(imageList[idxImage])
        else:
            testSet.append(imageList[idxImage])

    return trainSet, testSet


def normalize_data(trainSet, testSet):

    for idxTrain in range(len(trainSet)):
        currentImage = trainSet[idxTrain]
        maxValue = np.amax(currentImage)
        maxValue = maxValue.astype(float)
        normalizedImage = [x.astype(float)/maxValue for x in currentImage]
        trainSet[idxTrain] = normalizedImage

    for idxTest in range(len(testSet)):
        currentImage = testSet[idxTest]
        maxValue = np.amax(currentImage)
        maxValue = maxValue.astype(float)
        currentImage = [x.astype(float) / maxValue for x in currentImage]
        testSet[idxTest] = currentImage

    return trainSet, testSet


def eliminate_nonRGB(imageList, categoryList):
    copy_imageList_asArray = []
    copy_categoryList = []
    for idxImage in range(len(imageList)):
        currentShape = np.shape(imageList[idxImage])
        is_3D = (len(currentShape) == 3)
        if is_3D:
            thirdD_is_3 = (currentShape[2] == 3)
        else:
            thirdD_is_3 = False
        if thirdD_is_3:
            copy_imageList_asArray.append(imageList[idxImage])
            copy_categoryList.append(categoryList[idxImage])
    imageList = copy_imageList_asArray
    categoryList = copy_categoryList

    return imageList, categoryList


def zeroPad2Square(imageList):
    for idxImage in range(len(imageList)):
        currentImage = imageList[idxImage]
        height = np.size(currentImage, 0)
        width = np.size(currentImage, 1)
        diff = height - width
        if diff < 0:  # wider than high -> zero pad dim 0
            diff = 0 - diff
            addUp = np.ceil(np.true_divide(diff, 2))
            addUp = addUp.astype(int)
            addBottom = np.floor(np.true_divide(diff, 2))
            addBottom = addBottom.astype(int)
            updatedImage = np.pad(currentImage, ((addUp, addBottom), (0, 0), (0, 0)), 'constant',
                                  constant_values=(0, 0))
        elif diff > 0:
            addLeft = np.ceil(np.true_divide(diff, 2))
            addLeft = addLeft.astype(int)
            addRight = np.floor(np.true_divide(diff, 2))
            addRight = addRight.astype(int)
            updatedImage = np.pad(currentImage, ((0, 0), (addLeft, addRight), (0, 0)), 'constant',
                                  constant_values=(0, 0))

        imageList[idxImage] = updatedImage

    return imageList


def rescale_imageList(imageList, categoryList):
    # Todo: Current version has lots of hardcoded stuff
    # meanImageSize = 0
    # minImageSize = 299
    # maxImageSize = 300
    # a = 0
    # b = 0
    copy_imageList = []
    copy_categoryList = []
    for idxImage in range(len(imageList)):
        shape = np.shape(imageList[idxImage])
        size = shape[1]
        if size == 300:
            copy_imageList.append(imageList[idxImage])
            copy_categoryList.append(categoryList[idxImage])
        # if size < 300:
        #     a += 1
        # if size > 300:
        #     b += 1
        # if size <= minImageSize:
        #     minImageSize = size
        # if size > maxImageSize:
        #     maxImageSize = size
        # meanImageSize += size
        # if size == 299:
        #     imageList[idxImage] = rescale_image(imageList[idxImage], 300, 299)


    meanImageSize = 300 #meanImageSize / len(imageList)


    # for idxImage in range(len(imageList)):
    #     shape = np.shape(imageList[idxImage])
    #     size = shape[1]
    #     if not size == meanImageSize:
    #         if size < meanImageSize:
    #             minImageSize = size
    #         else:
    #             maxImageSize = size
    #         currentImage = imageList[idxImage]
    #         currentImage = rescale_image(currentImage, meanImageSize, size)
    #         imageList[idxImage] = currentImage
    imageList = copy_imageList
    categoryList = copy_categoryList

    return imageList, categoryList, meanImageSize


def rescale_image(image, meanSize=300, currentSize=299):
    # ToDo: implement with specifics
    return_image = np.pad(image,((1,0),(1,0),(0,0)),'constant')
    return return_image


def preprocess_imageList(ground_data_path="/Users/henrik.mettler/Desktop/DataSetForAutoencoder"):

    # create a list of images from data set
    imageList, categoryList = create_imageList(ground_data_path)
    # clean image list from black and white images
    imageList, categoryList = eliminate_nonRGB(imageList, categoryList)
    # zero pad images to square size
    imageList = zeroPad2Square(imageList)

    # up-/down-sample image size to mean size
    imageList, categoryList, mean_imageSize = rescale_imageList(imageList, categoryList)

    return imageList, categoryList, mean_imageSize


def zeroPad2OutputSize(data, dimOutput):
    data_output = np.zeros(shape=[data.shape[0],dimOutput,dimOutput,data.shape[3]])
    dimInput = data.shape[1]
    numPadDim = (dimOutput - dimInput)/2
    isZero = 'false'
    # ToDo: this does the job, but is very ugly (and presuambly slow...)
    for a in range(data.shape[0]):
        data_output[a,:,:,:] = np.pad(data[a,:,:,:],((numPadDim,numPadDim), (numPadDim,numPadDim), (0, 0)), 'constant')

    return data_output


def create_sample_representation(encoder_Model, data, label, is_training):
    # pick up two random elements from the data set and their labels
    size_data = label.size
    element_one = np.random.randint(0,size_data-1)
    element_two = np.random.randint(0,size_data-1)
    label_one = label[element_one]
    label_two = label[element_two]

    # pick up the data from the element one
    data_one = data[element_one]
    picked_data = np.zeros((2, np.shape(data_one)[0], np.shape(data_one)[1], np.shape(data_one)[2]))
    picked_data[0,:,:,:] = data_one

    # check that they belong to different classes, if not repeat replacing the 2nd until the labels are no longer equal
    if label_one == label_two:
        label_are_equal = 'true'
    else:
        label_are_equal = 'false'

    while label_are_equal == 'true':
        element_two = np.random.randint(0,size_data-1)
        label_two = label[element_two]
        if label_one != label_two:
            label_are_equal = 'false'

    # pick up the data for the second element
    data_two = data[element_two]
    picked_data [1,:,:,:] = data_two

    # calculate the hidden representation of the two data inputs
    hidden_representations_beforereshape = encoder_Model.predict(picked_data)
    # reshape the hidden representations
    hidden_rep_1 = hidden_representations_beforereshape[0, :, :, :]
    hidden_rep_1 = hidden_rep_1.reshape([1, np.shape(hidden_rep_1)[0] * np.shape(hidden_rep_1)[1] * np.shape(hidden_rep_1)[2]])
    hidden_rep_2 = hidden_representations_beforereshape[1, :, :, :]
    hidden_rep_2 = hidden_rep_2.reshape([1, np.shape(hidden_rep_2)[0] * np.shape(hidden_rep_2)[1] * np.shape(hidden_rep_2)[2]])
    # put into one array
    hidden_representations = np.zeros((2, np.shape(hidden_rep_1,)[1]))
    hidden_representations[0, :] = hidden_rep_1
    hidden_representations[1, :] = hidden_rep_2

    # return the labels as a vector
    labels_out = [label_one, label_two]

    return hidden_representations, labels_out # explicitly don't return the data, as it is not accessible to the DM!


def create_dm_network(num_units, input_shape, optimizer, loss, output_activation='sigmoid'):

    # hidden_size = 128

    lstm_layer = LSTM(num_units, return_sequences=True, input_shape=(1, input_shape))

    dm_network = Sequential()
    dm_network.add(lstm_layer)
    dm_network.add(Dense(1,activation=output_activation))
    dm_network.compile(optimizer=optimizer, loss=loss)
    dm_network.summary()

    return dm_network


def create_networks(num_units, input_shape, optimizer, loss, output_activation='sigmoid', time_step_per_trial = 1):
    lstm_layer = LSTM(num_units,return_sequences=True, input_shape=(time_step_per_trial, input_shape))

    action_network = Sequential()
    action_network.add(lstm_layer)
    action_network.add(Dense(1,activation=output_activation))
    action_network.compile(optimizer=optimizer, loss=loss)
    action_network.summary()

    value_network = Sequential()
    value_network.add(lstm_layer)
    value_network.add(Dense(1,activation=output_activation))
    value_network.compile(optimizer=optimizer, loss=loss)
    value_network.summary()

    return action_network, value_network


def mix_up(hidden_rep1, hidden_rep2, labels):
    label1 = labels[0]
    label2 = labels[1]
    #copy_labels = labels
    p = np.random.random_sample()

    hidden_representations = np.zeros((2, np.size(hidden_rep1)))
    # change the order if p > 0.5
    if p < 0.5:
        hidden_representations[0, :] = hidden_rep1
        hidden_representations[1, :] = hidden_rep2
    else:
        hidden_representations[0, :] = hidden_rep2
        hidden_representations[1, :] = hidden_rep1
        labels[0] = label2
        labels[1] = label1

    return hidden_representations, labels


def trial_run(dm_network, hidden_representation, labels):
    # Run a trial: DM takes two hidden representations, previous rewards and previous actions as input and outputs an
    # action (selection of one of the two representations)
    # concatenate the two hidden reps into one input
    rep1 = hidden_representation[0,:,:,:]
    rep2 = hidden_representation[1,:,:,:]
    rep1 = rep1.reshape(np.shape(rep1)[0]*np.shape(rep1)[1]*np.shape(rep1)[2])
    rep2 = rep2.reshape(np.shape(rep2)[0] * np.shape(rep2)[1] * np.shape(rep2)[2])
    dm_input = np.concatenate((rep1, rep2))
    dm_input = np.array([dm_input])
    dm_input = np.array([dm_input])
    prediction = dm_network.predict(dm_input, batch_size=None, verbose=0, steps=None)
    choice = int(round(prediction)) # todo: change this rounding to be part of the model!

    return choice, prediction, dm_input


def check_reward(choice, labels, good_label):
    # convert choice to int if it is a float
    if isinstance(choice, float):
        choice = int(choice)

    if labels[choice] == good_label:
        reward = 1
    else:
        reward = -1

    return reward


def training(dm_network, target, prediction, choice, all_hidden_reps, optimizer, learning_rate):

    #loss = np.square(target - prediction)
    target = np.array([target])
    target = np.array([target])
    target = np.moveaxis(target,2,0)
    info = dm_network.fit(all_hidden_reps, target, verbose=0)
    loss = info.history
    return dm_network, loss


# Used to initialize weights for policy and value output layers
# def normalized_columns_initializer(std=1.0):
#     def _initializer(shape, dtype=None, partition_info=None):
#         out = np.random.randn(*shape).astype(np.float32)
#         out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
#         return tf.constant(out)
#
#     return _initializer


class Create_Args():
    # class for the implementation of an args data structure
    def __init__(self,nb_episodes, batch_size, consecutive_frames, training_interval, n_threads, gamma, lr, optimizer, n_timeSteps, gather_stats=0, render=0, env='empty'):
        self.nb_episode = nb_episodes
        self.batch_size = batch_size
        self.consecutive_frames = consecutive_frames
        self.training_interval = training_interval
        self.n_threads = n_threads
        self.gamma = gamma
        self.lr = lr
        self.optimizer = optimizer
        self.gather_stats = gather_stats
        self.render = render
        self.env = env
        self.n_timeSteps = n_timeSteps

def pick_digit_sample():
    hidden_reps = np.floor(10*np.random.rand(2)) # 2 random integers between 0,9
    while hidden_reps[0] == hidden_reps[1]: # replace the second if they are equal
        hidden_reps[1] = np.floor(10*np.random.random_sample())

    labels = np.zeros([2])
    labels[0] = np.int(hidden_reps[0])
    labels[1] = np.int(hidden_reps[1])

    return hidden_reps, labels
