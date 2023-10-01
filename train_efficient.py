import torch
from torch import nn
from unet.unet_transfer import UNet16, UNetResNet
from unet.efficient import EfficientUNet
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.autograd import Variable
import shutil
from data_loader import ImgDataSet
import os
import argparse
import tqdm
import numpy as np
import scipy.ndimage as ndimage
from torch.utils.tensorboard import SummaryWriter  

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_model(device, type ='vgg16'):
    if type == 'vgg16':
        print('create vgg16 model')
        model = UNet16(pretrained=True)
    elif type == 'resnet101':
        encoder_depth = 101
        num_classes = 1
        print('create resnet101 model')
        model = UNetResNet(encoder_depth=encoder_depth, num_classes=num_classes, pretrained=True)
    elif type == 'resnet34':
        encoder_depth = 34
        num_classes = 1
        print('create resnet34 model')
        model = UNetResNet(encoder_depth=encoder_depth, num_classes=num_classes, pretrained=True)
    # EfficientU-Net
    elif type == 'efficient':
        print('create efficient model')
        model = EfficientUNet()
    else:
        assert False
    model.eval()
    return model.to(device)

def adjust_learning_rate(optimizer, epoch, lr, adj_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = lr * (0.1 ** (epoch // 30))
    lr = lr * (0.1 ** (epoch // adj_lr))
    print(f"adj_lr is {adj_lr}")
    print(f"Learning rate is {lr}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

## %%
# def find_latest_model_path(dir, args):
#     model_paths = []
#     epochs = []
#     pretrained_path = []
#     for path in Path(dir).glob('*.pt'):
#         # 若路徑裡有'epoch'，則判斷有幾個，找出最新的epoch
#         # 若無，則跳出迴圈，進行下一個
#         if 'epoch' not in path.stem:
#             if 'model_unet_vgg_16_best' in path.stem and args.model_type=='vgg16':
#                 pretrained_path.append(path)
#             elif 'model_unet_res_net_best' in path.stem and (args.model_type=='resnet101' or args.model_type=='resnet34'):
#                 pretrained_path.append(path)
#             continue
#         model_paths.append(path)
#         parts = path.stem.split('_')
#         epoch = int(parts[-1])
#         epochs.append(epoch)
    
#     if len(epochs) > 0:
#         epochs = np.array(epochs)
#         max_idx = np.argmax(epochs)
#         return model_paths[max_idx]
#     elif len(pretrained_path) > 0:
#         return pretrained_path
#     else:
#         return None

def find_latest_model_path(dir):
    model_paths = []
    epochs = []
    for path in Path(dir).glob('*.pt'):
        if 'epoch' not in path.stem:
            continue
        model_paths.append(path)
        parts = path.stem.split('_')
        epoch = int(parts[-1])
        epochs.append(epoch)

    if len(epochs) > 0:
        epochs = np.array(epochs)
        max_idx = np.argmax(epochs)
        return model_paths[max_idx]
    else:
        return None



# def train(train_loader, model, criterion, optimizer, validation, args):
def train(train_loader, model, criterion, validation, args):

    # latest_model_path = find_latest_model_path(args.model_dir, args)
    latest_model_path = find_latest_model_path(args.model_dir)
    print(f"latest_model_path is {latest_model_path}")
    best_model_path = os.path.join(*[args.model_dir, 'model_best.pt'])
    print(f"best_model_path is {best_model_path}")


    if latest_model_path is not None:
        state = torch.load(latest_model_path)
        epoch = state['epoch']
        model.load_state_dict(state['model'])
        epoch = epoch

        #if latest model path does exist, best_model_path should exists as well
        assert Path(best_model_path).exists() == True, f'best model path {best_model_path} does not exist'
        #load the min loss so far
        best_state = torch.load(latest_model_path)
        min_val_los = best_state['valid_loss']

        print(f'Restored model at epoch {epoch}. Min validation loss so far is : {min_val_los}')
        epoch += 1
        print(f'Started training model from epoch {epoch}')
    else:
        print('Started training model from epoch 0')
        epoch = 0
        min_val_los = 9999
    

    # SGD, Momentum, AdaGrad, Adam
    # RMSProp: Efficient-Net
    # if(args.model_type=='efficient'):
    #     optimizer = torch.optim.RMSprop(model.parameters(),
    #                                     args.lr,
    #                                     momentum=args.momentum,
    #                                     weight_decay=args.weight_decay)
    # else:
    optimizer = torch.optim.SGD(model.parameters(), 
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    valid_losses = []

    for epoch in range(epoch, args.n_epoch + 1):

        adjust_learning_rate(optimizer, epoch, args.lr, args.adj_lr)

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}')

        losses = AverageMeter()

        model.train()

        # tensorboard顯示用的變數
        input_var = torch.zeros([args.batch_size, 3, args.input_size_H, args.input_size_W])
        masks_pred = torch.zeros([args.batch_size, 1, args.input_size_H, args.input_size_W])

        # train_loader (len is data's index)
        for i, (input, target) in enumerate(train_loader):
            # Why Variable? Not necessary！
            input_var  = Variable(input).cuda()
            target_var = Variable(target).cuda()
            # y->output = f(x->input)
            masks_pred = model(input_var)
            # print(f"masks_pred is {masks_pred.shape}")
            # 照片訓練完後，跟ground truth做loss運算
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat  = target_var.view(-1)
            loss = criterion(masks_probs_flat, true_masks_flat)
            # losses is AverageMeter()
            # def update(self, val, n=1):
            #     self.val = val
            #     self.sum += val * n
            #     self.count += n
            #     self.avg = self.sum / self.count
            losses.update(loss)
            # 顯示
            tq.set_postfix(loss='{:.5f}'.format(losses.avg))
            tq.update(args.batch_size)
            # print(tq.update(args.batch_size))

        # exit()
            # writer.add_image("Training data", input_var, tq.update(args.batch_size))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            
        print(f"\ninput_var is {input_var.shape}")
        print(f"masks_pred is {masks_pred.shape}")
        # writer.add_image("Training data", input_var[0], epoch)
        # writer.add_image("Ground truth", target_var[0], epoch)
        # writer.add_image("Prediction", masks_pred[0], epoch)

        # print(f"input_var is {input_var.shape}")
        # input_var_visual = np.reshape(input_var[0:i], (-1, 28, 28, 1))
        # print(f"input_var_visual is {input_var_visual.shape}")
        # print(f"masks_pred is {masks_pred.shape}")

        # visualize the images
        # writer.add_image("Training data", input_var, epoch)
        # writer.add_image("Training data", masks_pred, epoch)
        # visualize the loss
        # writer.add_scalar("loss", loss.detach(), epoch)
        # writer.add_scalar("loss", loss, epoch)


        # 寫在epoch的for迴圈裡面
        # writer.add_image(f"train_tfms {train_tfms}")
        # writer.add_image(f"val_tfms {val_tfms}")
        # writer.add_image(f"mask_tfms {mask_tfms}")

        # validation
        valid_metrics = validation(model, valid_loader, criterion)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        print(f'\tvalid_loss = {valid_loss:.5f}')
        tq.close()

        # writer.add_scalar("validation_loss", valid_loss.detach(), epoch)
        # writer.add_scalar("validation_loss", valid_loss, epoch)

        # All losses
        # writer.add_scalars("Losses", {"Training Loss":loss,
        #                              "Validation Loss": valid_loss}, epoch)

        #save the model of the current epoch
        epoch_model_path = os.path.join(*[args.model_dir, f'model_epoch_{epoch}.pt'])
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'valid_loss': valid_loss,
            'train_loss': losses.avg
        }, epoch_model_path)

        if valid_loss < min_val_los:
            min_val_los = valid_loss

            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_loss': valid_loss,
                'train_loss': losses.avg
            }, best_model_path)

def validate(model, val_loader, criterion):
    losses = AverageMeter()
    model.eval()
    with torch.no_grad():

        for i, (input, target) in enumerate(val_loader):
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            output = model(input_var)
            loss = criterion(output, target_var)

            losses.update(loss.item(), input_var.size(0))

    return {'valid_loss': losses.avg}

def save_check_point(state, is_best, file_name = 'checkpoint.pth.tar'):
    torch.save(state, file_name)
    if is_best:
        shutil.copy(file_name, 'model_best.pth.tar')

def calc_crack_pixel_weight(mask_dir):
    avg_w = 0.0
    n_files = 0
    for path in Path(mask_dir).glob('*.*'):
        n_files += 1
        m = ndimage.imread(path)
        ncrack = np.sum((m > 0)[:])
        w = float(ncrack)/(m.shape[0]*m.shape[1])
        avg_w = avg_w + (1-w)

    avg_w /= float(n_files)

    return avg_w / (1.0 - avg_w)

if __name__ == '__main__':
    # Change parameter
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-n_epoch', default=10, type=int, metavar='N', help='number of total epochs to run')
    # EfficientU-Net 1e-5 -lr 0.00001
    parser.add_argument('-lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-adj_lr', default=30, type=int, help='adjust learning rate every X epoch')
    parser.add_argument('-momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('-print_freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-batch_size',  default=4, type=int,  help='weight decay (default: 1e-4)')
    parser.add_argument('-num_workers', default=4, type=int, help='output dataset directory')
    # Tensor - CHW
    # EfficientU-Net: 640, 648
    parser.add_argument('-input_size_H', default=448, type=int, help='input data image size')
    parser.add_argument('-input_size_W', default=448, type=int, help='input data image size')
    parser.add_argument('-log_dir', default='logs', type=str, help='save log')
    # Important
    parser.add_argument('-data_dir',type=str, help='input dataset directory')
    parser.add_argument('-model_dir', type=str, help='output dataset directory')
    parser.add_argument('-model_type', type=str, required=False, default='resnet101', choices=['vgg16', 'resnet101', 'resnet34', 'efficient'])


    args = parser.parse_args()
    # 創資料夾 exist_ok
    os.makedirs(args.model_dir, exist_ok=True)

    # if(args.model_type=='vgg16'):
    # writer = SummaryWriter(f"{args.log_dir}/{args.model_type}")
    # if(args.model_type=='resnet101'):
    #     writer = SummaryWriter(args.model_type+"/"+args.model_type)
    # if(args.model_type=='resnet34'):
    #     writer = SummaryWriter(args.model_type+"/"+args.model_type)
    # if(args.model_type=='efficient'):
    #     writer = SummaryWriter(args.model_type+"/"+args.model_type)

    # Get dataset
    DIR_IMG  = os.path.join(args.data_dir, 'images')
    DIR_MASK = os.path.join(args.data_dir, 'masks')

    img_names  = [path.name for path in Path(DIR_IMG).glob('*.jpg')]
    mask_names = [path.name for path in Path(DIR_MASK).glob('*.jpg')]
    print(f'total images = {len(img_names)}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # writer.add_image(f"img_names {img_names}")
    # writer.add_image(f"mask_names {mask_names}")

    # exit()

    model = create_model(device, args.model_type)


    # # SGD, Momentum, AdaGrad, Adam
    # # RMSProp: Efficient-Net
    # if(args.model_type=='efficient'):
    #     optimizer = torch.optim.RMSprop(model.parameters(), args.lr,
    #                                     momentum=args.momentum,
    #                                     weight_decay=args.weight_decay)
    # else:
    #     optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                                 momentum=args.momentum,
    #                                 weight_decay=args.weight_decay)

    #crack_weight = 0.4*calc_crack_pixel_weight(DIR_MASK)
    #print(f'positive weight: {crack_weight}')
    #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([crack_weight]).to('cuda'))
    criterion = nn.BCEWithLogitsLoss().to('cuda')
    
    # Data preproccess (for normalize)
    # 全部data的RGB平均值/255
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]

    # 原圖進來轉tensor和做normalize，train完一次epoch後，
    # 做調整亮度、對比度、旋轉與翻轉等..
    # epoch越多，資料改得越多次
    train_tfms = transforms.Compose([transforms.ToTensor(),
                                     # Try to enlarge
                                     transforms.Resize([args.input_size_H, args.input_size_W]),
                                     # contrast: 對比度 saturation: 飽和度 hue: 色調
                                    #  transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                    #  transforms.RandomRotation(degrees=360),
                                    #  transforms.RandomCrop(448),
                                    #  transforms.RandomHorizontalFlip(p=0.5),
                                    #  transforms.RandomVerticalFlip(p=0.5),
                                    #  transforms.GaussianBlur(3),
                                     transforms.Normalize(channel_means, channel_stds)])

    val_tfms = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize([args.input_size_H, args.input_size_W]),
                                   transforms.Normalize(channel_means, channel_stds)])

    mask_tfms = transforms.Compose([transforms.ToTensor(),
                                    # transforms.RandomRotation(degrees=360),
                                    # transforms.RandomHorizontalFlip(p=0.5),
                                    # transforms.RandomVerticalFlip(p=0.5),
                                    transforms.Resize([args.input_size_H, args.input_size_W], transforms.InterpolationMode.NEAREST)])


    # DIR_IMG(資料位置)  img_names(資料)  train_tfms(資料transform)
    # DIR_MASK(答案位置) mask_names(答案) mask_tfms(答案transform)
    dataset = ImgDataSet(img_dir=DIR_IMG, img_fnames=img_names, img_transform=train_tfms, mask_dir=DIR_MASK, mask_fnames=mask_names, mask_transform=mask_tfms)
    # 85% training data ; 15% validation
    train_size = int(0.85*len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    print(f"\ntrain_dataset is {len(train_dataset)}")
    print(f"valid_dataset is {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)

    # for i, (input, target) in enumerate(train_loader):
    #     input_var  = input.cuda()
    #     target_var = target.cuda()
    #     writer.add_image("Training data", input_var[0], i)
    #     writer.add_image("Validation", target_var[0], i)


    model.cuda()

    # train(train_loader, model, criterion, optimizer, validate, args)
    train(train_loader, model, criterion, validate, args)

    # writer.close()