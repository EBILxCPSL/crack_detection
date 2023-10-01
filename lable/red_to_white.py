import numpy as np
import os
from PIL import Image
from pathlib import Path
import cv2 as cv

np.set_printoptions(threshold=np.inf)

def RedToWhite(img_dir, new_img_dir):
    folders = os.listdir(img_dir)
    for floder in folders:
        image_path = os.path.join(img_dir, floder)
        # img = Image.open(image_path)
        img = cv.imread(image_path)
        print(f"img is {img.shape}")
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         pixel = img[i][j]
        #         if not(pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0): #BGR if not black
        #             img[i][j] = np.array([255, 255, 255])


        black_pixels_mask = np.all(img == [0, 0, 0], axis=-1)

        # non_black_pixels_mask = np.any(img != [0, 0, 0], axis=-1)  
        # or non_black_pixels_mask = ~black_pixels_mask

        # image_copy[black_pixels_mask] = [255, 255, 255]
        img[~black_pixels_mask] = [255, 255, 255]
        # newImg = np.array(img)
        # newImg
        # newImg = np.clip(img, 0, 255)
        # print(f"newImg is {newImg.shape}")
        # newImg = newImg.astype(np.uint8)
        # newImg = Image.fromarray(img)
        # print(f"newImg is {newImg.size}")
        newImg_path = os.path.join(new_img_dir,floder)
        cv.imwrite(newImg_path, img)
        print(newImg_path)
        # newImg.save(newImg_path)

if __name__ == '__main__':
    img_path = 'red_masks_v2'
    newImg_path = 'masks_v3'
    RedToWhite(img_path, newImg_path)
