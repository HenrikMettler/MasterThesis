import os
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
    for idxFolder in range(len(path_list)):
        fileList = os.listdir(path_list[idxFolder])
        numberFiles = len(fileList)
        label = idxFolder + 1

        for idxFile in range(numberFiles):
            fileString = brain_path + '/' + fileList[idxFile]
            currentImage = Image.open(fileString,'r')
            currentImage_data = currentImage.load()
            currentImage_object = ImageObject(currentImage_data, label)
            imageList.append(currentImage_object)

    return imageList

