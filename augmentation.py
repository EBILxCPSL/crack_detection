import os
import torch
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

def plot_images(x, y, images):
    assert len(images) <= x * y
    f, ax = plt.subplots(x, y)
    
    idx = 0
    if x == 1:
        for j in range(y):
            if idx < len(images):
                ax[j].imshow(images[idx])
                idx += 1
            else:
                break
        
    else:
        for i in range(x):
            for j in range(y):
                if idx < len(images):
                    ax[i, j].imshow(images[idx])
                    idx += 1
                else:
                    break

    plt.show()
    return

if __name__ == '__main__':
    DATA_PATH = ".\\dataset_v4\\images"
    MASK_PATH = ".\\dataset_v4\\masks"
    # DATA_PATH = ".\\dataset\\images"
    # MASK_PATH = ".\\dataset\\masks"
    data_names = []
    data_fnames = []
    mask_names = []
    mask_fnames = []
    dataset = []
    # 讀取images和masks的路徑
    for images in os.listdir(DATA_PATH):
        if "augment" in str(images):
            continue
        data_fnames.append(os.path.join(DATA_PATH, images))
        images = Path(images).stem
        data_names.append(os.path.join(DATA_PATH, images))

    for masks in os.listdir(MASK_PATH):
        if "augment" in str(masks):
            continue
        mask_fnames.append(os.path.join(MASK_PATH, masks))
        masks = Path(masks).stem
        mask_names.append(os.path.join(MASK_PATH, masks))

        # if os.path.isdir(os.path.join(DATA_PATH, images)) and not images.startswith("."):
        #     for data in os.listdir(os.path.join(DATA_PATH, images)):
    #             if os.path.isdir(os.path.join(DATA_PATH, images, data)) and not data.startswith("."):
    #                 if "augment" in str(data):
    #                     continue
    #                 data_names.append(os.path.join(DATA_PATH, images, data))
    
    data_names = sorted(data_names)
    data_fnames = sorted(data_fnames)
    mask_names = sorted(mask_names)
    mask_fnames = sorted(mask_fnames)
    dataset = list(zip(data_names, mask_names))
    dataset_fullname = list(zip(data_fnames, mask_fnames))
    # # print(f"dataset is {dataset}")


    data_fnames_aug = []
    mask_fnames_aug = []
    n = 1
    for data_path in dataset:
        # data_path[0] -> images
        # data_path[1] -> masks
        # print(f"data_path[0] is {data_path[0]}")
        # print(f"data_path[1] is {data_path[1]}")
    #     aug_path = data_path[0]+"_augment"
    #     if not os.path.isdir(aug_path):
    #         os.makedirs(aug_path)
    #     image_paths = [os.path.join(data_path, data) for data in os.listdir(data_path) if data.endswith(".jpg")]
        for n in range(1, 21):
            img = data_path[0].split('\\')[-1] + f"_augment{n}.jpg"
            msk = data_path[1].split('\\')[-1] + f"_augment{n}.jpg"
            # print(f"img is {img}")
            # print(f"msk is {msk}")

            # images
            original_img = data_path[0] + ".jpg"
            target_img = os.path.join(DATA_PATH, img)
            data_fnames_aug.append(target_img)
            # print(f"original_img is {original_img}")
            # print(f"target_img is {target_img}")
            shutil.copyfile(original_img, target_img)
            
            # masks
            original_msk = data_path[1] + ".jpg"
            target_msk = os.path.join(MASK_PATH, msk)
            mask_fnames_aug.append(target_msk)
            # print(f"original_msk is {original_msk}")
            # print(f"target_msk is {target_msk}")
            shutil.copyfile(original_msk, target_msk)

            # if os.path.isfile(os.path.join(DATA_PATH, img)):
            #     index = img.find('ori')
            #     aug_or = img[:index] + 'augment_' + img[index:]
            #     original = os.path.join(DATA_PATH, img)
                # target = os.path.join(aug_path, aug_gt)
                # print(f'original: {original}')
                # print(f'target: {target}')
                # shutil.copyfile(original, target)
        #     else:
        #         print(f'Cannot find gt at {os.path.join(data_path, data)}\nfile_dir: {data_path}\tfile_name: {data}')
        #         break
            
        # print(f"data_fnames_aug is {data_fnames_aug}")
        # print(f"mask_fnames_aug is {mask_fnames_aug}")

    MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])
    for path in data_fnames_aug:
        # min: -0.02    max: 0.08
        color_shift = 0.3 * np.random.random(3) - 0.02 # color shift for all channel
        # min: 0.9      max: 1.1
        brightness = 1 + 0.2 * np.random.random(3) - 0.15 # brightness change for all channel
        # print(f"path is {path}")
        tsi = np.array(Image.open(path)) / 255
        aug = tsi + MEAN_COLOR_RGB
        aug *= brightness
        aug += color_shift
        aug -= MEAN_COLOR_RGB
        aug *= 255
        aug = np.clip(aug, 0, 255).astype(int)
        im = Image.fromarray(np.uint8(aug))
        file_name = path.split('\\')[-1]
        print(f"file_name is {file_name}")
        im.save(os.path.join(DATA_PATH, file_name))
        # index = file_name.find('.')
        # aug_file = file_name[:index-1] + '_1' + file_name[index-1:]
        # print(f"aug_file is {aug_file}")
        # im.save(os.path.join(DATA_PATH, aug_file))
#         im.save(os.path.join(aug_path, aug_file))
# #         images = [tsi, aug]
# #         plot_images(1, 2, images)
#     print(f'Saved at {aug_path}')