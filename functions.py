import os
import numpy as np
# import pickle

from PIL import Image
from create_imageObject import *

def create_imageList():
    # path to all data
    ground_data_path = "/Users/henrik.mettler/Desktop/DataSetForAutoencoder"
    # folder paths
    brain_path = os.path.join(ground_data_path, "brain")
    buddha_path = os.path.join(ground_data_path, "buddha")
    cellphone_path = os.path.join(ground_data_path, "cellphone")
    crocodile_path = os.path.join(ground_data_path, "crocodile")
    dolphin_path = os.path.join(ground_data_path, "dolphin")
    helicopter_path = os.path.join(ground_data_path, "helicopter")
    laptop_path = os.path.join(ground_data_path, "laptop")
    pizza_path = os.path.join(ground_data_path, "pizza")
    revolver_path = os.path.join(ground_data_path, "revolver")
    sunflower_path = os.path.join(ground_data_path, "sunflower")

    path_list = [brain_path, buddha_path, cellphone_path, crocodile_path, dolphin_path, helicopter_path, laptop_path,
                 pizza_path, revolver_path, sunflower_path]

    imageList = []
    imageList_asArray = []
    for idxFolder in range(len(path_list)):
        fileList = os.listdir(path_list[idxFolder])
        numberFiles = len(fileList)
        label = idxFolder + 1

        for idxFile in range(numberFiles):
            fileString = brain_path + '/' + fileList[idxFile]
            currentImage = Image.open(fileString,'r')

            currentImage_asArray = np.asarray(currentImage)
            imageList_asArray.append(currentImage_asArray)

            # currentImage_data = currentImage.load()
            # currentImage_object = ImageObject(currentImage_data, label)
            # imageList.append(currentImage_object)

    return imageList_asArray


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


def eliminate_nonRGB(imageList):
    copy_imageList_asArray = []
    for idxImage in range(len(imageList)):
        currentShape = np.shape(imageList[idxImage])
        is_3D = (len(currentShape) == 3)
        if is_3D:
            thirdD_is_3 = (currentShape[2] == 3)
        else:
            thirdD_is_3 = False
        if thirdD_is_3:
            copy_imageList_asArray.append(imageList[idxImage])
    imageList = copy_imageList_asArray

    return imageList


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


def rescale_imageList(imageList):
    meanImageSize = 0
    minImageSize = 300
    maxImageSize = 300
    for idxImage in range(len(imageList)):
        shape = np.shape(imageList[idxImage])
        size = shape[1]
        meanImageSize += size
    meanImageSize = meanImageSize / len(imageList)


    for idxImage in range(len(imageList)):
        shape = np.shape(imageList[idxImage])
        size = shape[1]
        a = 3
        if not size == meanImageSize:
            if size < meanImageSize:
                minImageSize = size
            else:
                maxImageSize = size
            currentImage = imageList[idxImage]
            currentImage = rescale_image(currentImage, meanImageSize, size)
            imageList[idxImage] = currentImage
    return imageList, meanImageSize


def rescale_image(image_asArray, meanSize, currentSize):
    # ToDo: implement
    raise NameError('not yet implemented (not necessary in 1st dataset')
    return image_asArray


def preprocess_imageList():

    # create a list of images from data set
    imageList = create_imageList()
    # clean image list from black and white images
    imageList = eliminate_nonRGB(imageList)
    # zero pad images to square size
    imageList = zeroPad2Square(imageList)

    # up-/down-sample image size to mean size
    imageList, mean_imageSize = rescale_imageList(imageList)

    return imageList, mean_imageSize
