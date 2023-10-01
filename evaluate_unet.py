from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_score, recall_score

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

def binary_pa(s, g):
    """
        calculate the pixel accuracy of two N-d volumes.
        s: the segmentation volume of numpy array
        g: the ground truth volume of numpy array
        """
    pa = ((s == g).sum()) / g.size
    return pa

# def mpa(y_true, y_pred):
#     y_true, y_pred = y_true.flatten(), y_pred.flatten()
#     FP = len(np.where(y_pred - y_true == 1)[0])
#     FN = len(np.where(y_true - y_pred == 1)[0])
#     TP = len(np.where(y_pred + y_true == 2)[0])
#     TN = len(np.where(y_pred + y_true == 0)[0])

#     pos = (TP + 1e-15) / (TP + FN + 1e-15)
#     neg = (TN + 1e-15) / (TN + FP + 1e-15)

#     return (pos+neg)/2

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
    arg = parser.add_argument
    arg('-ground_truth_dir', type=str, default='.\ground_truth', required=False, help='path where ground truth images are located')
    arg('-pred_dir', type=str, default='.\\test_result', required=False,  help='path with predictions')
    arg('-threshold', type=float, default=0.3, required=False,  help='crack threshold detection')
    args = parser.parse_args()

    result_mpa = []
    result_bin_mpa = []
    result_miou = []
    result_dice = []
    result_jaccard = []
    result_recall = []

    paths = [path for path in  Path(args.ground_truth_dir).glob('*')]
    for file_name in tqdm(paths):
        y_true = (cv2.imread(str(file_name), 0) > 0).astype(np.uint8)

        # unique, counts = np.unique(y_true, return_counts=True)
        # gt_cnt = dict(zip(unique, counts))
        # print(f"gt_cnt is {gt_cnt}")

        pred_file_name = Path(args.pred_dir) / file_name.name
        if not pred_file_name.exists():
            print(f'missing prediction for file {file_name.name}')
            continue

        y_pred = (cv2.imread(str(pred_file_name), 0) > 255 * args.threshold).astype(np.uint8)

        # print(y_true.max(), y_true.min())
        # plt.subplot(131)
        # plt.imshow(y_true)
        # plt.subplot(132)
        # plt.imshow(y_pred)
        # plt.subplot(133)
        # plt.imshow(y_true)
        # plt.imshow(y_pred, alpha=0.5)
        # plt.show()

        result_mpa += [mpa(y_true, y_pred)]
        result_bin_mpa += binary_pa(y_true, y_pred)
        result_jaccard += [jaccard(y_true, y_pred)]
        result_dice += [dice(y_true, y_pred)]
        result_recall += [recall(y_true, y_pred)]

    print('mPA = ', np.mean(result_mpa), np.std(result_mpa))
    print('mIoU = ', np.mean(result_jaccard), np.std(result_jaccard))
    print('Dice = ', np.mean(result_dice), np.std(result_dice))
    print('Recall = ', np.mean(result_recall), np.std(result_recall))