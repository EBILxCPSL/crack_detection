import os
import torch
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


if __name__ == '__main__':
    DATA_PATH = ".\label_v2"
    json_list = os.listdir(DATA_PATH)
    for i in json_list:
        if '.json' in i:
            json = os.path.join(DATA_PATH, i)
            print(f"json is {json}")
            os.system(f"labelme_json_to_dataset {json}")


    # json_list = os.listdir(DATA_PATH)
    # json = glob.glob('*.json')
    # file = glob.glob('*_json')
    # check = False
        # for j in file:
        #     i = i.split('.')[0]
        #     j = j.split('_')[0]+j.split('_')[1]+j.split('_')[2]+j.split('_')[3]
        #     print(f"i is {i}")
        #     print(f"j is {j}")
        #     if(i!=j):
        #         print("Not this")
        #         continue
        #     else:
        #         check = True
        #         print("Exist")    
        # if(check==True):
        #     continue
        # else:
        #     print(f"Create: {i}.json")
        #     os.system(f"labelme_json_to_dataset {i}.json")