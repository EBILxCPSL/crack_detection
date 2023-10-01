import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.pyplot import MultipleLocator

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-model_dir", type=str, required=True)
    ap.add_argument("-title", type=str, required=True)
    ap.add_argument("-ylabel", default='Loss',  type=str, help='label the y-axis')
    ap.add_argument("-xlabel",default='Epochs', type=str, help='label the x-axis')
    args = ap.parse_args()

    paths = [path for path in Path(args.model_dir).glob('*.pt')]
    paths = sorted(paths)
    epochs = []
    tr_losses = []
    vl_losses = []
    for path in tqdm(paths):
        if 'epoch' not in path.stem:
            continue
        if 'current' in path.stem:
            continue
        #load the min loss so far
        parts = path.stem.split('_')
        epoch = int(parts[-1])
        epochs.append(epoch)
        state = torch.load(path)
        val_los = state['valid_loss']
        train_loss = float(state['train_loss'])
        tr_losses.append(train_loss)
        vl_losses.append(val_los)

    epochs = np.sort(epochs)
    print(f"epochs is {epochs}")
    sorted_idxs = np.argsort(epochs)
    print(f"sorted_idxs is {sorted_idxs}")
    tr_losses = [tr_losses[idx] for idx in sorted_idxs]
    vl_losses = [vl_losses[idx] for idx in sorted_idxs]

    # plt.xticks(ticks=tr_losses, labels=epochs)
    # plt.xticks(ticks=vl_losses, labels=epochs)

    plt.xlim(-10, epochs[-1]+10)
    # plt.plot(tr_losses[1:], label='train_loss')
    # plt.plot(vl_losses[1:], label='valid_loss')
    plt.plot(epochs, tr_losses, label='train_loss')
    plt.plot(epochs, vl_losses, label='valid_loss')
    plt.title(args.title)   # title: {Model_name}
    plt.ylabel(args.ylabel) # y    : Loss
    plt.xlabel(args.xlabel) # x    : Epochs
    plt.legend()
    plt.show()


