from __future__ import print_function
from argparse import ArgumentParser
import torchvision
from torchvision.models import VGG16_Weights
import os
from pathlib import Path
import glob
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

# parser = ArgumentParser()
# # parser.add_argument("pos1", help="positional argument 1")
# # parser.add_argument("pos2", help="positional argument 1")
# parser.add_argument("-o", "--optional-arg", help="optional argument", dest="opt", default="default")
# args = parser.parse_args()

# # print("positional1 arg:", args.pos1)
# # print("positional2 arg:", args.pos2)
# print("optional arg:", args.opt)

# encoder = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1) #.features
# print(encoder)

# path = 'C:\\Users\\EBILxCPSL_S3\\Desktop\\Thesis\\7.Cracks\\0.Model\\Keyu\\dataset\\CrackForest-dataset-master'
# get_img = os.path.join(path, 'image')
# print(get_img)
# img = [path.name for path in Path(get_img).glob('*.jpg')]
# print(len(img))


# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label
    
#     def __len__(self):
#         return len(self.img_labels)
    
# print(torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features)

# class Student:
#     def __init__(self, name, gender, dep, ID):
#         self.name = name
#         self.gender = gender
#         self.dep = dep
#         self.ID = ID

# class Person(Student):
#     def __init__(self, name, gender, dep, ID, pro_qua):
#         super().__init__(name, gender, dep, ID)
#         self.name = name
#         self.gender = gender
#         self.dep = dep
#         self.ID = ID
#         self.pro_qua = pro_qua
#     def profess(self):
#         g = {'c', 'c++', 'Java'}
#         s = {'python', 'R'}
#         e = {'Mat.', 'Fortran'}

#         if self.pro_qua in g:
#             return g
#         elif self.pro_qua in s:
#             return s
#         else: return e

# student_A = Person('Keyu', 'Male', 'EECS', '39832012', 'python')
# print(student_A.name)
# print(student_A.gender)
# print(student_A.dep)
# print(student_A.profess(), student_A.pro_qua)

# ResNet
# print(torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
#                      'nvidia_efficientnet_b4',
#                       pretrained=True))

# path = os.getcwd("F:\\Bridge Inspection\\Dataset\\images")
# path = "F:\\Bridge Inspection\\Dataset\\images"
# path = 'C:\\Users\\EBILxCPSL_S3\\Desktop\\Crack_detection\\label\images\\新增資料夾'
path = r'C:\Users\EBILxCPSL_S3\Desktop\Crack_detection\label\masks'
images = glob.glob(path+'/*')
# dirlist= os.listdir(path)
# (todofolder,foldername)= os.path.split(path)  
# dirlist= os.listdir(path)
n=1
for i in images:
    print(i)
    # os.rename(i, path+f'/Touqian_River_Bridge_{n:03d}.jpg')
    # os.rename(i, path+f'/County_Highway_117_{n:03d}.jpg')
    os.rename(i, path+f'/crack_{n:03d}.jpg')
    n = n+1