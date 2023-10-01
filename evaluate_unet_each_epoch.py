from pathlib import Path
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_loader import ImgDataSet
from unet.unet_transfer import UNet16, input_size
from utils import load_unet_vgg16, load_unet_resnet_101, load_unet_resnet_34
# from resunet_test import resunet34 , ResUnet , BasicBlock
from unet.unet_transfer import UNet16, UNetResNet, EfficientUNet

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os
import json
from sklearn.metrics import jaccard_score
import warnings

def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def general_dice(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return dice(y_true, y_pred)

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def general_jaccard(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return jaccard(y_true, y_pred)

def mpa(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    FP = len(np.where(y_pred - y_true == 1)[0])
    FN = len(np.where(y_true - y_pred == 1)[0])
    TP = len(np.where(y_pred + y_true == 2)[0])
    TN = len(np.where(y_pred + y_true == 0)[0])

    pos = (TP + 1e-15) / (TP + FN + 1e-15)
    neg = (TN + 1e-15) / (TN + FP + 1e-15)

    return (pos+neg)/2

def miou(y_true, y_pred):
    return jaccard_score(y_true.flatten(), y_pred.flatten(), average='macro')


def evaluate_img(model, img, transform):
    input_width, input_height = input_size[0], input_size[1]

    img_1 = cv.resize(img, (input_width, input_height), cv.INTER_AREA)
    X = transform(Image.fromarray(img_1))
    X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]

    mask = model(X)

    mask = torch.sigmoid(mask[0, 0]).data.cpu().numpy()
    mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)
    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-model_dir' ,default='./model_output', type=str, required=True)
    arg('-model_type', type=str, required=False, default='resnet101', choices=['vgg16', 'resnet101', 'resnet34', 'efficient'])
    arg('-dataset_jsonfile', type=str, default='./split_dataset/crack_dataset_split.json', required=True, help='path where ground truth images are located')
    arg('-threshold', type=float, default=0.2, required=False,  help='crack threshold detection')
    arg('-target_dir', type=str, default='./resUnet34', required=False, help=None)
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    epochs = []

    dice_list = []
    jaccard_list = []
    mpa_list = []
    miou_list = []

    valid_dataset = ImgDataSet(args.dataset_jsonfile, isTrain=False, img_transform=None, mask_transform=None)
    mskPaths = valid_dataset.maskList
    imgPaths = valid_dataset.imList
    dataloader = [list(i) for i in zip(imgPaths, mskPaths)]

    modelPaths = [path for path in Path(args.model_dir).glob('*.pt')]
    #print(modelPaths)
    modelPaths = sorted(modelPaths)

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]

    if not os.path.exists(os.path.join(args.target_dir, 'evaluate.json')):
        for i,modelPath in enumerate(modelPaths):
            if 'epoch' not in modelPath.stem:
                continue
            parts = modelPath.stem.split('_')
            epoch = int(parts[-1])
            epochs.append(epoch)

            tq = tqdm(total=len(dataloader))
            tq.set_description(f'Epoch {i}')

            if args.model_type == 'vgg16':
                model = load_unet_vgg16(modelPath)
            elif args.model_type  == 'resnet101':
                model = load_unet_resnet_101(modelPath)
            elif args.model_type  == 'resnet34':
                #model = load_unet_resnet_34(modelPath)
                # model = resunet34(3,64)
                model.cuda()
                print("每一次 modelpath" , modelPath)
                model.load_state_dict(torch.load(modelPath)['model'])
                print(model)
                
            else:
                print('undefind model name pattern')
                exit()

            result_dice = []
            result_jaccard = []
            result_mpa = []
            result_miou = []
            
            for data in dataloader:
                img_pth, mask_pth = data[0], data[1]
                img_pth = Path(img_pth)
                mask_pth = Path(mask_pth)

                train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])

                img_0 = Image.open(str(img_pth))
                img_0 = np.asarray(img_0)
                if len(img_0.shape) != 3:
                    print(f'incorrect image shape: {img_pth.name}{img_0.shape}')
                    continue

                img_0 = img_0[:,:,:3]
                img_height, img_width, img_channels = img_0.shape
                prob_map_full = evaluate_img(model, img_0, train_tfms)
                y_pred = (prob_map_full > args.threshold).astype(np.int8)
                y_true = (cv.imread(str(mask_pth), 0) > 0).astype(np.int8)

                tq.update(1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result_dice += [dice(y_true, y_pred)]
                    result_jaccard += [jaccard(y_true, y_pred)]
                    result_mpa += [mpa(y_true, y_pred)]
                    result_miou += [miou(y_true, y_pred)]

            tq.close()

            dice_list.append(np.mean(result_dice))
            jaccard_list.append(np.mean(result_jaccard))
            mpa_list.append(np.mean(result_mpa))
            miou_list.append(np.mean(result_miou))

            print('Dice = ', np.mean(result_dice), np.std(result_dice))
            print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))
            print('MPA = ', np.mean(result_mpa), np.std(result_mpa))
            print('Miou = ', np.mean(result_miou), np.std(result_miou))

        sorted_idxs = np.argsort(epochs)
        print(f"{epochs=}")
        print(f"{sorted_idxs=}")
        dice_list = [dice_list[idx] for idx in sorted_idxs]
        jaccard_list = [jaccard_list[idx] for idx in sorted_idxs]
        mpa_list = [mpa_list[idx] for idx in sorted_idxs]
        miou_list = [miou_list[idx] for idx in sorted_idxs]

        data = {}
        data['dice']=dice_list
        data['jaccard']=jaccard_list
        data['mpa']=mpa_list
        data['miou']=miou_list
        with open(os.path.join(args.target_dir, 'evaluate.json'), 'w', newline='') as jsonfile:
            json.dump(data, jsonfile)
    else:
        with open(os.path.join(args.target_dir, 'evaluate.json')) as jsf:
            data_load = json.load(jsf)
        dice_list = data_load['dice']
        jaccard_list = data_load['jaccard']
        mpa_list = data_load['mpa']
        miou_list = data_load['miou']

    fig1, axis1 = plt.subplots()
    axis1.plot(dice_list)
    # plt.title(args.title)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    fig1.savefig(os.path.join(args.target_dir, 'avg_dice.jpg'), dpi=500)

    fig2, axis2 = plt.subplots()
    axis2.plot(miou_list)
    plt.title('Average MIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mean intersection over union')
    # plt.legend()
    fig2.savefig(os.path.join(args.target_dir, 'avg_miou.jpg'), dpi=500)

    fig3, axis3 = plt.subplots()
    axis3.plot(mpa_list)
    plt.title('Average MPA')
    plt.xlabel('Epoch')
    plt.ylabel('mean pixel accuracy')
    # plt.legend()
    fig3.savefig(os.path.join(args.target_dir, 'avg_mpa.jpg'), dpi=500)



