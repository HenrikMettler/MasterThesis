import os
import numpy as np
# import pickle

from PIL import Image
from create_imageObject import *

def create_imageList(ground_data_path):
    # path to all data
    ground_data_path = "/Users/henrik.mettler/Desktop/AutoencoderDataSet" # Adapt this to local path
    # folder paths
    path_list = os.listdir(ground_data_path)

    # brain_path = os.path.join(ground_data_path, "brain")
    # buddha_path = os.path.join(ground_data_path, "buddha")
    # cellphone_path = os.path.join(ground_data_path, "cellphone")
    # crocodile_path = os.path.join(ground_data_path, "crocodile")
    # dolphin_path = os.path.join(ground_data_path, "dolphin")
    # helicopter_path = os.path.join(ground_data_path, "helicopter")
    # laptop_path = os.path.join(ground_data_path, "laptop")
    # pizza_path = os.path.join(ground_data_path, "pizza")
    # revolver_path = os.path.join(ground_data_path, "revolver")
    # sunflower_path = os.path.join(ground_data_path, "sunflower")
    #
    # path_list = [brain_path, buddha_path, cellphone_path, crocodile_path, dolphin_path, helicopter_path, laptop_path,
    #              pizza_path, revolver_path, sunflower_path]

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