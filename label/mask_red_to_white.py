import numpy as np
import os
from PIL import Image

np.set_printoptions(threshold=np.inf)

def RedToWhite(img_dir,new_img_dir):
    folders = os.listdir(img_dir)
    for floder in folders:
        image_path = os.path.join(img_dir, floder)
        img = Image.open(image_path)
        newImg=np.array(img)*255
        newImg = newImg.astype(np.uint8)
        print(f"newImg is {newImg.shape}")
        newImg=Image.fromarray(newImg)
        newImg_path=os.path.join(new_img_dir,floder)
        newImg.save(newImg_path)

if __name__ == '__main__':
    img_path='.\\red_masks_v2'
    newImg_path='.\\masks_v2'
    RedToWhite(img_path, newImg_path)