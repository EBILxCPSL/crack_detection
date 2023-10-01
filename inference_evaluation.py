import sys
import os
import numpy as np
from pathlib import Path
import cv2 as cv
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from unet.unet_transfer import UNet16, input_size
import matplotlib.pyplot as plt
import argparse
from os.path import join
from PIL import Image
import gc
from utils import load_unet_vgg16, load_unet_resnet_101, load_unet_efficient
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, recall_score

def evaluate_img(model, img):
    # input_size為training時設定的大小
    input_width, input_height = input_size[0], input_size[1]
    
    # 用opencv做照片的resize
    img_1 = cv.resize(img, (input_width, input_height), cv.INTER_AREA)
    print(f"Resize {img.shape} to {img_1.shape}")
    
    # Array轉回PIL的 H x W x C
    img_1 = Image.fromarray(img_1)
    
    # 照片 -> totensor & normalize
    X_ = train_tfms(img_1)
    # 原本照片 3 x H x W -> 1 x 3 X H x W
    X = Variable(X_.unsqueeze(0)).cuda()  # [N, 1, H, W]
    print(f"Change {X_.shape} to {X.shape}")

    mask = model(X)

    mask = F.sigmoid(mask[0, 0]).data.cpu().numpy()
    mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)
    print(f"Result: {mask.shape}")
    
    return mask

def evaluate_img_patch(model, img):
    # input_size == arg.input_size 可以調整
    input_width, input_height = input_size[0], input_size[1]

    # input img
    img_height, img_width, img_channels = img.shape
    print(f"\nPatch img.shape is {img.shape}")

    # 若input img太小，則不做split
    if img_width < input_width or img_height < input_height:
        print("input img is too small, go to resize..")
        return evaluate_img(model, img)

    print(f"img_height is {img_height}")
    print(f"img_width is {img_width}")
    # stride = arg.input_size*0.1 int(448*0.1=44.8) -> 44
    if img_height==2156 and img_width==1840:
        stride_y = 539
        stride_x = 460
        print(f"stride_y is {stride_y}")
        print(f"stride_x is {stride_x}")
    elif img_height==1376 and img_width==1844:
        stride_y = 344
        stride_x = 461
        print(f"stride_y is {stride_y}")
        print(f"stride_x is {stride_x}")
    else:
        stride_ratio = 0.1
        stride_y = int(input_width * stride_ratio)
        stride_x = int(input_width * stride_ratio)
        print(f"stride_y is {stride_y}")
        print(f"stride_x is {stride_x}")

    # normalization_map 與 input img一樣大小
    normalization_map = np.zeros((img_height, img_width), dtype=np.int16)
    print(f"normalization_map is {normalization_map.shape}")

    patches = []
    patch_locs = []
    print("Split image into patch..")
    # y = (0, 2156-448+1, 44) = (0, 1709, 44) = 0, 44, 88, ...
    for y in range(0, img_height - input_height + 1, stride_y):
        for x in range(0, img_width - input_width + 1, stride_x):
            segment = img[y:y + input_height, x:x + input_width]
            # print(f"\ny:{y}~{y+input_height} x:{x}~{x+input_width} segment is {segment.shape}")
            normalization_map[y:y + input_height, x:x + input_width] += 1
            patches.append(segment)
            patch_locs.append((x, y))

    print("Get patches")
    patches = np.array(patches)
    # print(f"patches is {patches.shape}")
    if len(patch_locs) <= 0:
        return None
    # assert False
    preds = []
    for i, patch in enumerate(patches):
        patch_n = train_tfms(Image.fromarray(patch))
        X = Variable(patch_n.unsqueeze(0)).cuda()  # [N, 1, H, W]
        masks_pred = model(X)
        mask = F.sigmoid(masks_pred[0, 0]).data.cpu().numpy()
        preds.append(mask)

    probability_map = np.zeros((img_height, img_width), dtype=float)
    for i, response in enumerate(preds):
        coords = patch_locs[i]
        probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response

    # print(f"probability_map is {probability_map[0]}")
    # print(f"normalization_map is {normalization_map[0]}")
    # check = probability_map / normalization_map
    # print(f"probability_map / normalization_map is {check[0]}")   
    print("Finish combine!")

    return probability_map

def disable_axis():
    plt.axis('off')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.gca().axes.get_yaxis().set_ticklabels([])

def mpa(y_true, y_pred):
    # Ensure that both masks have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("Input masks must have the same shape.")

    # Calculate True Positives, True Negatives, False Positives, and False Negatives
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    return (TP + TN + 1e-15) / (TP + TN + FP + FN + 1e-15)

def binary_pa(y_true, y_pred):
    # Test
    return ((y_pred == y_true).sum()) / y_true.size


# The Jaccard Index, also known as the IoU
def jaccard(y_true, y_pred):
    # IoU = intersection / union
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


def dice(y_true, y_pred):
    # IoU = 2*intersection / SetA + SetB
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def general_dice(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0
    return dice(y_true, y_pred)

def recall(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return recall_score(y_true, y_pred, average='macro')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # inference: 用test data來評估模型好壞
    parser.add_argument('-img_dir',type=str, default='./test_imgs', required=False, help='input dataset directory')
    # 需跟training時的大小一樣
    parser.add_argument('-input_size', default=448, type=int, help='input data image size')
    parser.add_argument('-ground_truth_dir', type=str, default='./ground_truth', help='path where ground truth images are located')
    parser.add_argument('-model_path', type=str, help='trained model path')
    parser.add_argument('-model_type', type=str, required=False, default='efficient', choices=['vgg16', 'resnet101', 'resnet34', 'efficient'])
    parser.add_argument('-out_viz_dir', type=str, default='./test_result_viz', required=False, help='visualization output dir')
    parser.add_argument('-out_pred_dir', type=str, default='./test_result', required=False,  help='prediction output dir')
    parser.add_argument('-out_pred_dir_patch', type=str, default='./test_result_patch', required=False,  help='prediction output dir')
    parser.add_argument('-threshold', type=float, default=0.3 , help='threshold to cut off crack response')
    args = parser.parse_args()

    result_mpa_resize_list = []
    result_binary_pa_resize_list = []
    result_miou_resize_list = []
    result_dice_resize_list = []
    result_recall_resize_list = []

    result_mpa_patch_list = []
    result_binary_pa_patch_list = []
    result_miou_patch_list = []
    result_dice_patch_list = []
    result_recall_patch_list = []

    result_mpa_best_list = []
    result_binary_pa_best_list = []
    result_miou_best_list = []
    result_dice_best_list = []
    result_recall_best_list = []

    # mpa_matrix = []
    # binary_pa_matrix = []
    # miou_matrix = []
    # dice_matrix = []
    
    input_size = (args.input_size, args.input_size)

    # 若資料夾裡有資料 -> 刪除
    if args.out_viz_dir != '':
        os.makedirs(args.out_viz_dir, exist_ok=True)
        for path in Path(args.out_viz_dir).glob('*.*'):
            if 'checkpoints' in str(path):
                continue
            os.remove(str(path))

    # if args.out_viz_dir_patch != '':
    #     os.makedirs(args.out_viz_dir_patch, exist_ok=True)
    #     for path in Path(args.out_viz_dir_patch).glob('*.*'):
    #         if 'checkpoints' in str(path):
    #             continue
    #         os.remove(str(path))

    # if args.out_best_viz_dir != '':
    #     os.makedirs(args.out_best_viz_dir, exist_ok=True)
    #     for path in Path(args.out_best_viz_dir).glob('*.*'):
    #         if 'checkpoints' in str(path):
    #             continue
    #         os.remove(str(path))          

    if args.out_pred_dir != '':
        os.makedirs(args.out_pred_dir, exist_ok=True)
        for path in Path(args.out_pred_dir).glob('*.*'):
            if 'checkpoints' in str(path):
                continue
            os.remove(str(path))

    if args.out_pred_dir_patch != '':
        os.makedirs(args.out_pred_dir_patch, exist_ok=True)
        for path in Path(args.out_pred_dir_patch).glob('*.*'):
            if 'checkpoints' in str(path):
                continue
            os.remove(str(path))

    cm_path = f'{args.out_viz_dir}/confusion_matrix.txt'


    # Load .pt of the model
    if args.model_type == 'vgg16':
        model = load_unet_vgg16(args.model_path)
    elif args.model_type  == 'resnet101':
        model = load_unet_resnet_101(args.model_path)
    elif args.model_type  == 'efficient':
        model = load_unet_efficient(args.model_path)
        print(model)
    else:
        print('undefind model name pattern')
        exit()

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]

    ground_truth_dir = args.ground_truth_dir
    
    """ 讀取test_imgs裡的照片 """
    paths = [path for path in Path(args.img_dir).glob('*.*')]
    for path in tqdm(paths):
        #print(str(path))
        
        # 忽略JupyterLab在資料夾裡產生的.ipynb_checkpoints
        if 'checkpoints' in str(path):
            continue
        
        # torchvision resize的interpolation is BILINEAR
        train_tfms = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(channel_means, channel_stds)])

        # 讀取照片
        img_0 = Image.open(path)
        img_0 = np.asarray(img_0)
        print(f'\nimg_0 is {img_0.shape}')
        # 照片必須為 -> H x W x C
        if len(img_0.shape) != 3:
            print(f'incorrect image shape: {path.name}{img_0.shape}')
            continue
            
        # 確定照片為RGB -> 取照片的 img_height=H | img_width=W | img_channels=3
        img_0 = img_0[:,:,:3]
        img_height, img_width, img_channels = img_0.shape

        """ Resize the input image and get the output"""
        print("Resize and get output..")
        # 照片resize成arg.input_size後，丟進model做預測，得到output
        prob_map_full = evaluate_img(model, img_0)
        # 預測的照片存進test_result
        # filename is ./test_result/X.jpg
        print("Finish！ Save the resize predict image..")
        filename=join(args.out_pred_dir, f'{path.stem}.jpg')
        if args.out_pred_dir != '':
            cv.imwrite(filename, img=(prob_map_full * 255).astype(np.uint8))
        
        img_1 = img_0

        """ Split the input image into patch (if size < arg.input_size => resize) ,then put into model and get the output """
        # 照片切成patch做預測
        print("\nImage patch and get output..")
        prob_map_patch = evaluate_img_patch(model, img_1)
        filename_patch=join(args.out_pred_dir_patch, f'{path.stem}.jpg')
        print("Finish！ Save the patch predict image..")
        if args.out_pred_dir_patch != '':
            cv.imwrite(filename_patch, img=(prob_map_patch * 255).astype(np.uint8))

        plt.title(f'name={path.stem}. \n cut-off threshold = {args.threshold}', fontsize=4)
        prob_map_viz_patch = prob_map_patch.copy()
        prob_map_viz_patch = prob_map_viz_patch/ prob_map_viz_patch.max()
        prob_map_viz_patch[prob_map_viz_patch < args.threshold] = 0.0



        """ Evaluation resize's output """
        # 在ground_truth資料夾找與pred_img一樣名稱的照片    
        pred_img = f'{path.stem}.jpg'
        ground_truth_paths = [path for path in Path(ground_truth_dir).glob('*')]

        ground_truth_path = Path(f"{ground_truth_dir}/{pred_img}")
        # 判斷預測的照片檔名是否跟ground_truth裡的一樣
        if ground_truth_path.exists():

            print(f"Check -> pred_img:{pred_img} compares with ground_truth:{ground_truth_path.name}")

            # 讀取ground truth的masks，並且捨去RGB: H x W x C -> H x W
            y_true = (cv.imread(str(ground_truth_path), 0) > 0).astype(np.uint8)

            # 讀取resize predict的masks，並且捨去RGB: H x W x C -> H x W
            y_pred = (cv.imread(str(filename), 0) > 255 * args.threshold).astype(np.uint8)

            # 讀取patch predict的masks，並且捨去RGB: H x W x C -> H x W
            y_pred_patch = (cv.imread(str(filename_patch), 0) > 255 * args.threshold).astype(np.uint8)

            """ resize score """
            result_mpa = mpa(y_true, y_pred)
            result_binary_pa = binary_pa(y_true, y_pred)
            result_miou = jaccard(y_true, y_pred)
            result_dice = dice(y_true, y_pred)
            result_recall = recall(y_true, y_pred)
            result_mpa_resize_list += [result_mpa]
            result_binary_pa_resize_list += [result_binary_pa]
            result_miou_resize_list += [result_miou]
            result_dice_resize_list += [result_dice]
            result_recall_resize_list += [result_recall]

            """ patch score """
            result_mpa_patch = mpa(y_true, y_pred_patch)
            result_binary_pa_patch = binary_pa(y_true, y_pred)
            result_miou_patch = jaccard(y_true, y_pred_patch)
            result_dice_patch = dice(y_true, y_pred_patch)
            result_recall_patch = recall(y_true, y_pred_patch)
            result_mpa_patch_list += [result_mpa_patch]
            result_binary_pa_patch_list += [result_binary_pa_patch]
            result_miou_patch_list += [result_miou_patch]
            result_dice_patch_list += [result_dice_patch]
            result_recall_patch_list += [result_recall_patch]

            # y_true_flatten = y_true_cm.flatten()
            # y_pred_flatten = y_pred_cm.flatten()
            # y_pred_patch_flatten = y_pred_patch_cm.flatten()
            # y_true_flatten = list(y_true_flatten)
            # y_pred_flatten = list(y_true_flatten)
            # y_pred_patch_flatten = list(y_true_flatten)
            # print(f"y_true_flatten: {y_true_flatten}")
            # print(f"y_pred_flatten: {y_pred_flatten}")
            # print(f"y_pred_patch_flatten: {y_pred_patch_flatten}")
            # print(f"y_true.shape is {y_true.shape}")
            # print(f"y_true_flatten.shape is {y_true_flatten.shape}")
            # print(f"y_true.type is {y_true.dtype}")
            
            # print(f"y_pred.shape is {y_pred.shape}")
            # print(f"y_pred.type is {y_pred.dtype}")

            """ Best score """
            if result_mpa > result_mpa_patch:
                result_mpa_best_list += [result_mpa]
                result_mpa_best = result_mpa
                mpa_matrix = confusion_matrix(y_true.flatten(), y_pred.flatten())
                # recall = recall_score(y_true.flatten(), y_pred.flatten(), average='macro')
            else:
                result_mpa_best_list += [result_mpa_patch]
                result_mpa_best = result_mpa_patch
                # print(f"{ground_truth_path} y_true.shape: {y_true.shape}\n{y_true[0]}")
                # print(f"{filename} y_pred_patch.shape: {y_pred_patch.shape}\n{y_pred_patch[0]}")
                mpa_matrix = confusion_matrix(y_true.flatten(), y_pred_patch.flatten())
                # recall = recall_score(y_true.flatten(), y_pred.flatten(), average='macro')

            print(f"mpa_matrix is \n{mpa_matrix}")

            if result_binary_pa > result_binary_pa_patch:
                result_binary_pa_best_list += [result_binary_pa]
                result_binary_pa_best = result_binary_pa
                # binary_pa_matrix = confusion_matrix(y_true.flatten(), y_pred.flatten())
            else:
                result_binary_pa_best_list += [result_binary_pa_patch]
                result_binary_pa_best = result_binary_pa_patch
                # binary_pa_matrix = confusion_matrix(y_true.flatten(), y_pred_patch.flatten())

            if result_miou > result_miou_patch:
                result_miou_best_list += [result_miou]
                result_miou_best = result_miou
                # miou_matrix = confusion_matrix(y_true.flatten(), y_pred.flatten())
            else:
                result_miou_best_list += [result_miou_patch]
                result_miou_best = result_miou_patch
                # miou_matrix = confusion_matrix(y_true.flatten(), y_pred_patch.flatten())

            if result_dice > result_dice_patch:
                result_dice_best_list += [result_dice]
                result_dice_best = result_dice
                dice_matrix = confusion_matrix(y_true.flatten(), y_pred.flatten())
                # recall = recall_score(y_true.flatten(), y_pred.flatten(), average='macro')
            else:
                result_dice_best_list += [result_dice_patch]
                result_dice_best = result_dice_patch     
                dice_matrix = confusion_matrix(y_true.flatten(), y_pred_patch.flatten())
                # recall = recall_score(y_true.flatten(), y_pred.flatten(), average='macro')

            if result_recall > result_recall_patch:
                result_recall_best_list += [result_recall]
                result_recall_best = result_recall
            else:
                result_recall_best_list += [result_recall_patch]
                result_recall_best = result_recall_patch

            # print(mpa_matrix)
            with open(cm_path, 'a') as e:
                e.write(f'\n{path.stem}\n')
                # e.write(f'mpa_matrix: {mpa_matrix}\n')
                # e.write(f'binary_pa_matrix: {binary_pa_matrix}\n')
                # e.write(f'miou_matrix: {miou_matrix}\n')
                e.write(f'confusion_matrix: {dice_matrix}\n')


            truth = cv.imread(str(ground_truth_path), 0)


            
            # show the best dice score of result
            if result_dice > result_dice_patch:
                prob_map_viz_full = prob_map_full.copy()
            else:
                prob_map_viz_full = prob_map_patch.copy()

            prob_map_viz_full[prob_map_viz_full < args.threshold] = 0.0
            prob_map_viz_full[prob_map_viz_full > args.threshold] = 1.0
            prob_map_viz_full = prob_map_viz_full.squeeze() # HxW
            pred = torch.zeros([prob_map_viz_full.shape[0], prob_map_viz_full.shape[1], 3])
            H, W = prob_map_viz_full.shape
            # truth: 紅色 | pred: 藍色 | 預測正確: 白色
            for i in range(H):
                for j in range(W):
                    if int(prob_map_viz_full[i][j]) == 1:
                        pred[i][j][0] = 0
                        pred[i][j][1] = 0
                        pred[i][j][2] = 255 #blue

            ground_truth_bw = torch.zeros([truth.shape[0], truth.shape[1], 3])
            H, W = truth.shape                        
            for i in range(H):
                for j in range(W):
                    if int(truth[i][j]) == 255:
                        ground_truth_bw[i][j][0] = 255
                        ground_truth_bw[i][j][1] = 0
                        ground_truth_bw[i][j][2] = 0
                        
            assert H == truth.shape[0] and W == truth.shape[1]
            ground_truth = torch.zeros([truth.shape[0], truth.shape[1], 3])
            H, W = truth.shape
            for i in range(H):
                for j in range(W):
                    if int(truth[i][j]) == 255 and int(prob_map_viz_full[i][j]) == 1:
                        ground_truth[i][j][0] = 255
                        ground_truth[i][j][1] = 255 #green
                        ground_truth[i][j][2] = 255
                    elif int(truth[i][j]) == 255:
                        ground_truth[i][j][0] = 255 #red
                        ground_truth[i][j][1] = 0
                        ground_truth[i][j][2] = 0
                    elif int(prob_map_viz_full[i][j]) == 1:
                        ground_truth[i][j][0] = 0
                        ground_truth[i][j][1] = 0
                        ground_truth[i][j][2] = 255 #blue

            fig = plt.figure()

            st = fig.suptitle(f'name={path.stem} \n mPA = {result_mpa_best:.2f} mIoU = {result_miou_best:.2f} Dice = {result_dice_best:.2f}', fontsize="x-large")
            fig.subplots_adjust(top=1.3)

            # 原圖(左)
            ax = fig.add_subplot(141)
            ax.imshow(img_0)
            ax.axis('off')
            ax.set_title("Original Image")

            # predict圖(中)
            # prob_map_viz_full為H x W
            ax = fig.add_subplot(142)
            ax.imshow(ground_truth_bw)
            ax.axis('off')
            ax.set_title("Ground Truth")            
            
            # predict圖(中)
            # prob_map_viz_full為H x W
            ax = fig.add_subplot(143)
            ax.imshow(pred)
            ax.axis('off')
            ax.set_title("Predition")

            # mask'紅色' + predict'藍色'圖(右)
            ax = fig.add_subplot(144)
            ax.imshow(ground_truth)
            ax.axis('off')
            ax.set_title("Truth+Pred")

            # matrix = ConfusionMatrixDisplay(confusion_matrix=dice_matrix)
            # matrix.plot()
            # plt.show()
            # ax = fig.add_subplot(155)
            # ax.imshow(matrix)
            # ax.axis('off')



            plt.savefig(join(args.out_viz_dir, f'{path.stem}.jpg'), dpi=500)
            plt.close('all')
    
        else:
            print(f"{ground_truth_path} is not found！")

            
        gc.collect()
        # break
        
    print("\nOnly resize result:")
    print(f'mPA = {np.mean(result_mpa_resize_list)}, {np.std(result_mpa_resize_list)}')
    print(f'binary_PA = {np.mean(result_binary_pa_resize_list)}, {np.std(result_binary_pa_resize_list)}')
    print(f'mIoU = {np.mean(result_miou_resize_list)}, {np.std(result_miou_resize_list)}')
    print(f'Dice = {np.mean(result_dice_resize_list)}, {np.std(result_dice_resize_list)}')
    print(f'Recall = {np.mean(result_recall_resize_list)}, {np.std(result_recall_resize_list)}')

    print("\nOnly patch result:")
    print(f'mPA = {np.mean(result_mpa_patch_list)}, {np.std(result_mpa_patch_list)}')
    print(f'binary_PA = {np.mean(result_binary_pa_patch_list)}, {np.std(result_binary_pa_patch_list)}')
    print(f'mIoU = {np.mean(result_miou_patch_list)}, {np.std(result_miou_patch_list)}')
    print(f'Dice = {np.mean(result_dice_patch_list)}, {np.std(result_dice_patch_list)}')
    print(f'Recall = {np.mean(result_recall_patch_list)}, {np.std(result_recall_patch_list)}')

    print("\nBest result:")
    print(f'mPA = {np.mean(result_mpa_best_list)}, {np.std(result_mpa_best_list)}')
    print(f'binary_PA = {np.mean(result_binary_pa_best_list)}, {np.std(result_binary_pa_best_list)}')
    print(f'mIoU = {np.mean(result_miou_best_list)}, {np.std(result_miou_best_list)}')
    print(f'Dice = {np.mean(result_dice_best_list)}, {np.std(result_dice_best_list)}')
    print(f'Recall = {np.mean(result_recall_best_list)}, {np.std(result_recall_best_list)}')