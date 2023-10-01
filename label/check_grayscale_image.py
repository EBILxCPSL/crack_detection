import os
import cv2
from PIL import Image
from pathlib import Path

def is_grey_scale(img_path):
    img = Image.open(img_path).convert('RGB')
    img_name = Path(img_path).stem
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i,j))
            if r != g != b: 
                print(f"{img_name} is RGB image！")
                return False
    print(f"{img_name} is Grayscale image！")

def isgray(imgpath):
    img_name = Path(imgpath).stem
    img = cv2.imread(imgpath)
    if len(img.shape) < 3: print(f"{img_name} is Grayscale image！")
    if img.shape[2]  == 1: print(f"{img_name} is Grayscale image！")
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all(): print(f"{img_name} is Grayscale image！")
    else: print(f"{img_name} is RGB image！")

if __name__ == '__main__':
    # img_path='.\\red_masks'
    img_path=r"F:\Cracks\dataset_test\crop"
    folders = os.listdir(img_path)

    '''PIL'''
    for floder in folders:
        image_path = os.path.join(img_path, floder)
        is_grey_scale(image_path)

    '''cv2'''
    for floder in folders:
        image_path = os.path.join(img_path, floder)
        isgray(image_path)