import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random
from PIL import Image

# Load data
# preprocess (resize, pad, etc.)
# return images and masks
# DIR_IMG(資料位置)     img_names(資料)    train_tfms(資料transform) 
# DIR_MASK(答案位置)    mask_names(答案)   mask_tfms(答案transform)
class ImgDataSet(Dataset):
    # img_dir: The directory path where the original images are located.
    # img_fnames: A list of image filenames in img_dir.
    # img_transform: A transformation function to apply to each image.
    # mask_dir: The directory path where the mask images are located.
    # mask_fnames: A list of mask filenames in mask_dir.
    # mask_transform: A transformation function to apply to each mask.
    def __init__(self, img_dir, img_fnames, img_transform, mask_dir, mask_fnames, mask_transform):
        # images
        self.img_dir = img_dir
        self.img_fnames = img_fnames
        self.img_transform = img_transform
        # masks
        self.mask_dir = mask_dir
        self.mask_fnames = mask_fnames
        self.mask_transform = mask_transform

        self.seed = np.random.randint(2147483647)
    """
    This method is used to retrieve an item from the dataset at index i.
    It loads the image and mask corresponding to the index i,
    applies the specified transformations,
    and returns the transformed image and mask.
    """
    def __getitem__(self, i):
        fname = self.img_fnames[i]
        fpath = os.path.join(self.img_dir, fname)
        img = Image.open(fpath)
        if self.img_transform is not None:
            random.seed(self.seed)
            img = self.img_transform(img)
            #print('image shape', img.shape)

        mname = self.mask_fnames[i]
        mpath = os.path.join(self.mask_dir, mname)
        mask = Image.open(mpath)
        #print('khanh1', np.min(test[:]), np.max(test[:]))
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
            #print('mask shape', mask.shape)
            #print('khanh2', np.min(test[:]), np.max(test[:]))

        return img, mask, fpath #torch.from_numpy(np.array(mask, dtype=np.int64))

    def __len__(self):
        return len(self.img_fnames)

# 太佔記憶體空間
def padding(img,expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width //2
    pad_height = delta_height //2
    padding = (pad_width,pad_height,delta_width-pad_width,delta_height-pad_height)
    # return ImageOps.expand(img, padding)


"""
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
"""