import cv2
from pathlib import Path
import numpy as np
import os
from glob import glob

def cutimg(img, image_save_path_head, img_name):
    h,w, _ = img.shape
    print(f"h is {h}")
    print(f"w is {w}")
    # h = 1840/4 = 460
    if h % 4 == 0:
       cut1 = int(h/4)
    else:
        cut1 = int((h-1)/4)
    
    # w = 2156/4 = 539
    if w % 4 == 0:
       cut2 = int(w/4)
    else:
        cut2= int((w-1)/4)
    print('cut1=',cut1,'cut2=',cut2)

    #                 0~cut1 cut1~cut1*2 cut1*2~cut1*3 cut1*3~cut1*4
    #    0~cut2
    # cut2~cut2*2
    # cut2*2~cut2*3
    # cut2*3~cut2*4

    for i in range(0, 4):
         for j in range(0, 4):
            print(f"i*cut1 is {i*cut1}")
            print(f"(i+1)*cut1 is {(i+1)*cut1}")
            print(f"j*cut2 is {j*cut2}")
            print(f"(j+1)*cut2 is {(j+1)*cut2}")
            img_crop = img[i*cut1:(i+1)*cut1, j*cut2:(j+1)*cut2]
            crop_name = f"_crop{i+1}_{j+1}.jpg"
            cv2.imwrite(os.path.join(image_save_path_head, img_name + crop_name), img_crop)

    # img1_1 = img[0:cut1,0:cut2]
    # img1_2 = img[cut1:w,0:cut2]
    # img1_3 = img[0:cut1,cut2:h]
    # img1_4 =img[cut1:w,cut2:h]
    # cv2.imwrite(os.path.join(image_save_path_head, img_name+"_crop1.jpg"), img1_1)
    # cv2.imwrite(os.path.join(image_save_path_head, img_name+"_crop2.jpg"), img1_2)
    # cv2.imwrite(os.path.join(image_save_path_head, img_name+"_crop3.jpg"), img1_3)
    # cv2.imwrite(os.path.join(image_save_path_head, img_name+"_crop4.jpg"), img1_4)
    # img_list = {"img1_1":img1_1,"img1_2":img1_2,"img1_3":img1_3,"img1_4":img1_4}

def cutmsk(img, image_save_path_head, img_name):
    h,w = img.shape
    print(f"h is {h}")
    print(f"w is {w}")
    if h % 4 == 0:
       cut1 = int(h/4)
    else:
        cut1 = int((h-1)/4)
    
    if w % 4 == 0:
       cut2 = int(w/4)
    else:
        cut2= int((w-1)/4)
    print('cut1=',cut1,'cut2=',cut2)

    for i in range(0, 4):
        for j in range(0, 4):
            print(f"i*cut1 is {i*cut1}")
            print(f"(i+1)*cut1 is {(i+1)*cut1}")
            print(f"j*cut2 is {j*cut2}")
            print(f"(j+1)*cut2 is {(j+1)*cut2}")
            img_crop = img[i*cut1:(i+1)*cut1, j*cut2:(j+1)*cut2]
            crop_name = f"_crop{i+1}_{j+1}.jpg"
            cv2.imwrite(os.path.join(image_save_path_head, img_name + crop_name), img_crop)

if __name__ == '__main__':
    DATA_PATH = "./ground_truth_bridge_6_crop_all"
    MASK_PATH = "./ground_truth_bridge_6_crop_all"
    img_list = glob(os.path.join(DATA_PATH, "*.jpg"))
    for img in img_list:
        srcImg = cv2.imread(img)
        images = Path(img).stem
        if "crop" in images:
            continue
        print(f"img is {img}")
        print(f"srcImg is {srcImg.shape}")
        print(f"images is {images}")
        print(f"DATA_PATH is {DATA_PATH}")
        cutimg(srcImg, DATA_PATH, images)

    msk_list = glob(os.path.join(MASK_PATH, "*.jpg"))
    for msk in msk_list:
        srcMsk = cv2.imread(msk)
        grayMsk = cv2.cvtColor(srcMsk, cv2.COLOR_BGR2GRAY)
        masks = Path(msk).stem
        if "crop" in masks:
            continue
        print(f"msk is {msk}")
        print(f"srcMsk is {srcMsk.shape}")
        print(f"grayMsk is {grayMsk.shape}")
        print(f"masks is {masks}")
        print(f"MASK_PATH is {MASK_PATH}")
        cutmsk(grayMsk, MASK_PATH, masks)

    # # DATA_PATH = ".\\dataset\\images"
    # # MASK_PATH = ".\\dataset\\masks"
    # data_names = []
    # data_fnames = []
    # mask_names = []
    # mask_fnames = []
    # dataset = []
    # for images in os.listdir(DATA_PATH):
    #     data_fnames.append(os.path.join(DATA_PATH, images))
    #     print(f"data_fnames is {data_fnames}")
    #     images = Path(images).stem
    #     data_names.append(os.path.join(DATA_PATH, images))
    #     # srcImg = cv2.imread(data_fnames)
    #     # print(f"srcImg is {srcImg}")
    #     print(f"SAVE_PATH is {SAVE_PATH}")
    #     print(f"images is {images}")
    #     # cutimg(srcImg, SAVE_PATH, images)

    # for masks in os.listdir(MASK_PATH):
    #     mask_fnames.append(os.path.join(MASK_PATH, masks))
    #     masks = Path(masks).stem
    #     mask_names.append(os.path.join(MASK_PATH, masks))




# image_path = ".\dataset_test\images\crack_001.jpg"
# srcImg = cv2.imread(image_path)
# print(srcImg.shape)

# img_name = Path(image_path).stem
# print(img_name)

# image_save_path_head = ".\dataset_test\crop"

# img_fname = os.path.join(image_save_path_head, img_name)
# print(img_fname)




# cutimg(srcImg)

 
# cv2.namedWindow("[srcImg]", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("[srcImg]", srcImg)
# cv2.waitKey(0)
 


# image_save_path_tail = ".jpg"
# seq = 1
# for i in range(2):  # [1]480*360==15*11---height
#     for j in range(2):  # [2]column-----------width
#         img_roi = srcImg[(i * 112):((i + 1) * 112), (j * 112):((j + 1) * 112)]
#         image_save_path = "%s%d%s" % (image_save_path_head, seq, image_save_path_tail)##将整数和字符串连接在一起
#         cv2.imwrite(image_save_path, img_roi)
