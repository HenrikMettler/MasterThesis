import pickle
import seaborn as sns
import os


file_list=[]
dirpath = os.getcwd()
for file in os.listdir(dirpath + "/data"):
    if file.endswith(".pickle"):
        file_list.append(file)

this_dir, this_filename = os.path.split(__file__)
data_list = []
for file in file_list:
    file_path = os.path.join("data", file)
    with open(file_path, "rb") as f:
        while True:
            #try:
            current_data = pickle.load(f)#[idx]
            data_list.append(current_data)
            # except EOFError:
            #     print 'Pickle ends'
            #     break


