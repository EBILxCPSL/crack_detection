import shutil
import os
import random


if __name__ == '__main__':
    ROOT_DIR = os.path.abspath("../")
    # 從此資料夾傳出檔案
    img_path = os.path.join(ROOT_DIR, "cracks\\test_imgs_open_82")
    mask_path = os.path.join(ROOT_DIR, "cracks\\ground_truth_open_82")
    # 目的地
    copy_to_path = os.path.join(ROOT_DIR, "cracks\\r")
    copy_to_path_mask = os.path.join(ROOT_DIR, "cracks\\r")
    imglist = os.listdir(img_path)
    print(f"imglist is {len(imglist)}")
    # random_imglist = random.sample(imglist, int(0.2*len(imglist)))
    random_imglist = random.sample(imglist, int(240))
    print(f"random_imglist is {len(random_imglist)}")

    for img in random_imglist:
        mask_path_file = os.path.join(mask_path, img)

        if os.path.exists(mask_path_file):
            # print(f"copy {os.path.join(mask_path, img)} to {os.path.join(copy_to_path_mask, img)}")
            shutil.copy(os.path.join(mask_path, img), os.path.join(copy_to_path_mask, img))
            os.remove(os.path.join(mask_path, img))
        # print(f"copy {os.path.join(img_path, img)} to {os.path.join(copy_to_path, img)}")
        shutil.copy(os.path.join(img_path, img), os.path.join(copy_to_path, img))
        os.remove(os.path.join(img_path, img))