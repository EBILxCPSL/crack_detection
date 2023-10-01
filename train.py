import torch
from torch import nn
from unet.unet_transfer import UNet16, UNetResNet, EfficientUNet
# from unet.efficient import EfficientUNet
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.autograd import Variable
from evaluate_unet import mpa, miou, dice, jaccard
import shutil
from data_loader import ImgDataSet
import os
import argparse
import tqdm
import numpy as np
import json
import scipy.ndimage as ndimage
from torch.utils.tensorboard import SummaryWriter  

class AverageMeter(object):
    """ Computes and stores the average and current value """
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

# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)
        
#         # flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()                            
#         dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
#         return 1 - dice


# class DiceLoss(nn.Module):
#     def __init__(self, p=1, smooth=1):
#         super(DiceLoss, self).__init__()
#         self.p = p
#         self.smooth = smooth
#     def forward(self, pred, target):
#         probs = torch.sigmoid(pred)
#         numer = (probs*target).sum()
#         denor = (probs.pow(self.p) + target.pow(self.p)).sum()
#         loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
#         return loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


def create_model(device, type ='vgg16'):
    """ 可以用torchsummary輸入照片大小去看model每層的feature maps """
    if type == 'vgg16':
        print('create vgg16 model')
        model = UNet16(pretrained=True)
    elif type == 'resnet101':
        encoder_depth = 101
        num_classes = 1
        print('create resnet101 model')
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
    """ Sets the learning rate to the initial LR decayed by 10 every 30 epochs """
    # lr = lr * (0.1 ** (epoch // 30))
    lr = lr * (0.1 ** int(epoch / adj_lr))
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
def train(train_loader, model, criterion, validation, args, model_output_dir):
    # freeze wieght一開始一定要先給權重，這邊是方便下指令
    # Command: -finetune_final_layer(不需打-initial_load)
    if args.finetune_final_layer is True:
        args.initial_load = True

    """
    判斷是否有之前還沒train好的epoch
                or
    若要load pre-trained weights -> Command: -initial_load
                or
    Initialize all the param
    """
    current_model_path = os.path.join(*[model_output_dir, 'model_epoch_current.pt'])
    if os.path.isfile(current_model_path):
        print(f"\nload current model: {current_model_path}")
        best_model_path = os.path.join(*[model_output_dir, 'model_best.pt'])
        print(f"best_model_path: {best_model_path}")
        state = torch.load(current_model_path)
        epoch = state['epoch'] + 1
        model.load_state_dict(state['model'])
        #if latest model path does exist, best_model_path should exists as well
        assert Path(best_model_path).exists() == True, f'best model path {best_model_path} does not exist'
        #load the min loss so far
        best_state = torch.load(current_model_path)
        min_val_los = best_state['valid_loss']
        print(f'Restored model at epoch {epoch}. Min validation loss so far is : {min_val_los}')
        print(f'Started training model from epoch {epoch}')

    elif args.initial_load:
        pretrained_path = ''
        for path in Path('pretrained_weights').glob('*.pt'):
            if 'model_unet_vgg_16_best' in path.stem and args.model_type=='vgg16':
                pretrained_path = os.path.join(*['pretrained_weights', 'model_unet_vgg_16_best.pt'])
            elif 'model_unet_res_net_best' in path.stem and (args.model_type=='resnet101' or args.model_type=='resnet34'):
                pretrained_path = os.path.join(*['pretrained_weights', 'model_unet_res_net_best.pt'])
            elif 'model_unet_efficient_net_best' in path.stem and (args.model_type=='efficient'):
                pretrained_path = os.path.join(*['pretrained_weights', 'model_unet_efficient_net_best.pt'])
        epoch = 0
        min_val_los = 9999
        best_model_path = os.path.join(*[model_output_dir, 'model_best.pt'])
        # load pt 裡面的權重進model
        state = torch.load(pretrained_path)
        model.load_state_dict(state['model'])
        print(f"\nload pretrain model: {pretrained_path}")
    else:
        # latest_model_path = find_latest_model_path(args.model_dir, args)
        latest_model_path = find_latest_model_path(model_output_dir)
        print(f"latest_model_path is {latest_model_path}")
        best_model_path = os.path.join(*[model_output_dir, 'model_best.pt'])
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
    

    # 使全部權重參數都不做gradient計算，只finetune final layer的參數
    """ Freeze the pretrained weight, and train other layers """
    if args.finetune_final_layer is True:
        for param in model.parameters():
            param.requires_grad = False

        """
        model.{裡面定義每層的變數}.{conv}.weight.requires_grad
                            + 
        model.{裡面定義每層的變數}.{conv}.bias.requires_grad
        """
        # model.dec2.block[1].conv.weight.requires_grad = True
        # model.dec2.block[1].conv.bias.requires_grad = True
        # model.dec2.block[2].conv.weight.requires_grad = True
        # model.dec2.block[2].conv.bias.requires_grad = True
        # model.dec1.conv.weight.requires_grad = True
        # model.dec1.conv.bias.requires_grad = True
        model.final.weight.requires_grad = True
        model.final.bias.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]

    # SGD, Momentum, AdaGrad, Adam
    # Efficient-Net -> 鈞陽建議用RMSProp (train不起來)
    # if(args.model_type=='efficient'):
    #     optimizer = torch.optim.RMSprop(
    #                                     param_dicts,
    #                                     # model.parameters(),
    #                                     args.lr,
    #                                     momentum=args.momentum,
    #                                     weight_decay=args.weight_decay)
    # else:
    if args.opti=='sgd':
        optimizer = torch.optim.SGD(param_dicts,
                                    # model.parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.opti=='adam':
        optimizer = torch.optim.Adam(param_dicts,
                                    # model.parameters(),
                                    args.lr,
                                    weight_decay=args.weight_decay)

    valid_losses = []

    for epoch in range(epoch, args.n_epoch+1):

        adjust_learning_rate(optimizer, epoch, args.lr, args.adj_lr)

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}')

        losses = AverageMeter()

        model.train()

        # tensorboard顯示用的變數
        input_var = torch.zeros([args.batch_size, 3, args.input_size_H, args.input_size_W])
        masks_viz = torch.zeros([args.batch_size, 1, args.input_size_H, args.input_size_W])
        # target_var = torch.zeros([args.batch_size, 1, args.input_size_H, args.input_size_W])
        # loss = torch.zeros([])

        # train_loader (len is data's index)
        for i, (input, target, path) in enumerate(train_loader):
            input_var  = Variable(input).cuda()
            target_var = Variable(target).cuda()
            # y- > output = f(x->input)
            masks_pred = model(input_var)

            # tensorboard shows prediction
            masks_viz = F.softmax(masks_pred, dim=1)
            # print(f"masks_pred is {masks_pred.shape}")

            # 照片訓練完後，跟ground truth做loss運算
            if args.loss=='bce':
                masks_probs_flat = masks_pred.view(-1)
                true_masks_flat  = target_var.view(-1)
                # criterion = nn.BCEWithLogitsLoss().to('cuda')
                loss = criterion(masks_probs_flat, true_masks_flat)
            elif args.loss=='dice':
                loss = criterion(masks_pred, target_var)

            # Reset and compute current value
            losses.update(loss)

            tq.set_postfix(loss='{:.5f}'.format(losses.avg))
            tq.update(args.batch_size)
            # print(args.batch_size)

            # Initialize, compute backpropagation(loss) and gradients, update the model's parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        """ Tensorboard """
        print(f"\ninput_var.shape is {input_var.shape}")
        print(f"masks_viz.shape is {masks_viz.shape}")
        # print(f"masks_viz.unique is {masks_viz.unique}")
        writer.add_image("Training data", input_var[0], epoch)
        writer.add_image("Ground truth", target_var[0], epoch)
        writer.add_image("Prediction", masks_viz[0]*255, epoch)

        # print(f"input_var is {input_var.shape}")
        # input_var_visual = np.reshape(input_var[0:i], (-1, 28, 28, 1))
        # print(f"input_var_visual is {input_var_visual.shape}")
        # print(f"masks_pred is {masks_pred.shape}")

        # visualize the images
        # writer.add_image("Training data", input_var, epoch)
        # writer.add_image("Training data", masks_pred, epoch)
        # visualize the loss
        # writer.add_scalar("loss", loss.detach(), epoch)
        writer.add_scalar("loss", loss, epoch)


        # 寫在epoch的for迴圈裡面
        # writer.add_image(f"train_tfms {train_tfms}")
        # writer.add_image(f"val_tfms {val_tfms}")
        # writer.add_image(f"mask_tfms {mask_tfms}")

        # validation
        valid_metrics = validation(model, valid_loader, criterion, epoch)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        print(f'\tvalid_loss = {valid_loss:.5f}')
        tq.close()

        # writer.add_scalar("validation_loss", valid_loss.detach(), epoch)
        writer.add_scalar("validation_loss", valid_loss, epoch)

        # All losses
        writer.add_scalars("Losses", {"Training Loss":loss,
                                     "Validation Loss": valid_loss}, epoch)

        # save the model of current epoch
        current_epoch_model_path = os.path.join(*[model_output_dir, f'model_epoch_current.pt'])
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'valid_loss': valid_loss,
            'train_loss': losses.avg,
            # 'argparse': args
        }, current_epoch_model_path)

        # save the model of every 20 epoch
        if epoch % args.save_every_epoch == 0:
            #save the model of the current epoch
            epoch_model_path = os.path.join(*[model_output_dir, f'model_epoch_{epoch}.pt'])
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



def validate(model, val_loader, criterion, epoch):
    result_mpa = []
    result_miou = []
    result_dice = []
    result_jaccard = []
    record_mpa_matrics = f'logs/{args.model_type}/experiment_{args.exp}/mpa.txt'
    record_miou_matrics = f'logs/{args.model_type}/experiment_{args.exp}/miou.txt'
    record_dice_matrics = f'logs/{args.model_type}/experiment_{args.exp}/dice.txt'
    record_jaccard_matrics = f'logs/{args.model_type}/experiment_{args.exp}/jaccard.txt'

    losses = AverageMeter()
    model.eval()
    # To not compute gradients or update model parameters
    with torch.no_grad():
        for i, (input, target, path) in enumerate(val_loader):
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            # if epoch == 0:
            #     print(f"Valid_image is {path}")

            output = model(input_var)
            loss = criterion(output, target_var)

            # Evaluation matrics input     '''output_eval = output.cpu().detach().numpy() or output.detach().cpu().numpy() ???'''   detach() -> return 新的tensor，requires_grad為false
            output_eval = output.cpu().detach().numpy().astype(np.uint8)
            target_eval = target_var.cpu().detach().numpy().astype(np.uint8)
            # print(f"output_eval is {output_eval.dtype}")
            # print(f"target_eval is {target_eval.dtype}")
            result_mpa += [mpa(output_eval, target_eval)]
            result_miou += [miou(output_eval, target_eval)]
            result_dice += [dice(output_eval, target_eval)]
            result_jaccard += [jaccard(output_eval, target_eval)]

            # .item() -> 從tensor取值      .size(0) -> 第0維度的資料
            losses.update(loss.item(), input_var.size(0))

        mean_mpa = np.mean(result_mpa)
        std_mpa = np.std(result_mpa)
        mean_miou = np.mean(result_miou)
        std_miou = np.std(result_miou)
        mean_dice = np.mean(result_dice)
        std_dice = np.std(result_dice)
        mean_jaccard = np.mean(result_jaccard)
        std_jaccard = np.std(result_jaccard)
        
        with open(record_mpa_matrics, 'a') as e:
            e.write(f'Epoch {epoch}\n')
            e.write(f'Mean: {mean_mpa}\tStd: {std_mpa}\n')

        with open(record_miou_matrics, 'a') as e:
            e.write(f'Epoch {epoch}\n')
            e.write(f'Mean: {mean_miou}\tStd: {std_miou}\n')

        with open(record_dice_matrics, 'a') as e:
            e.write(f'Epoch {epoch}\n')
            e.write(f'Mean: {mean_dice}\tStd: {std_dice}\n')

        with open(record_jaccard_matrics, 'a') as e:
            e.write(f'Epoch {epoch}\n')
            e.write(f'Mean: {mean_jaccard}\tStd: {std_jaccard}\n')

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
    parser.add_argument('-n_epoch', default=50, type=int, metavar='N', help='number of total epochs to run')
    # EfficientU-Net 1e-5 -lr 0.00001       pytorch_efficient -> defualt: 1e-2
    parser.add_argument('-lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    # No adjust lr -> adj_lr > n_epoch
    parser.add_argument('-adj_lr', default=7777, type=int, help='adjust learning rate every X epoch')
    parser.add_argument('-momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('-print_freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-batch_size',  default=4, type=int,  help='weight decay (default: 1e-4)')
    parser.add_argument('-num_workers', default=4, type=int, help='output dataset directory')
    # Tensor - CHW
    parser.add_argument('-input_size_H', default=448, type=int, help='input data image size')
    parser.add_argument('-input_size_W', default=448, type=int, help='input data image size')
    # Important
    parser.add_argument('-cuda', default=0, type=int, help='which cuda to use')
    parser.add_argument('-data_dir',type=str, help='input dataset directory')
    parser.add_argument('-model_dir', type=str, default='model_output', help='output dataset directory')
    parser.add_argument('-model_type', type=str, required=False, default='vgg16', choices=['vgg16', 'resnet101', 'efficient'])
    parser.add_argument('-initial_load', default=False, help='load pretrained weights', action='store_true')
    parser.add_argument('-finetune_final_layer', default=False, help='finetune_final_layer(No need -initial_load)', action='store_true')
    parser.add_argument('-loss', type=str, required=False, default='bce', choices=['bce', 'dice', 'dicebce'])
    parser.add_argument('-opti', type=str, required=False, default='sgd', choices=['sgd', 'adam'])
    # Choose one command
    parser.add_argument('-sep', default=False, help='separate valid imgs', action='store_true')
    parser.add_argument('-split', default=0.8, type=float, help='split the dataset into train and validation')
    # Save
    parser.add_argument('-save_every_epoch', default=10, type=int, help='save the model every X epoch')
    parser.add_argument('-exp', default=1, type=str, required=True, help='save the experiment logs')

    args = parser.parse_args()


    os.makedirs("logs", exist_ok=True)
    os.makedirs(f"logs/{args.model_type}", exist_ok=True)

    writer = SummaryWriter(f"logs/{args.model_type}/experiment_{args.exp}")
    model_output_dir = f"logs/{args.model_type}/experiment_{args.exp}/{args.model_dir}"
    # 創資料夾 exist_ok
    os.makedirs(model_output_dir, exist_ok=True)

    # Record the commandline
    with open(f'logs/{args.model_type}/experiment_{args.exp}/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    valid_dataset_path = f'{args.data_dir}\\validation'

    # Get dataset
    DIR_IMG  = os.path.join(args.data_dir, 'images')
    DIR_MASK = os.path.join(args.data_dir, 'masks')
    VALID_DIR_IMG  = os.path.join(valid_dataset_path, 'images')
    VALID_DIR_MASK = os.path.join(valid_dataset_path, 'masks')

    img_names  = [path.name for path in Path(DIR_IMG).glob('*.jpg')]
    mask_names = [path.name for path in Path(DIR_MASK).glob('*.jpg')]
    valid_img_names  = [path.name for path in Path(VALID_DIR_IMG).glob('*.jpg')]
    valid_mask_names = [path.name for path in Path(VALID_DIR_MASK).glob('*.jpg')]


    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): print(f"Using cuda")
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
    # Dice Loss
    if args.loss=='bce':
        criterion = nn.BCEWithLogitsLoss().to('cuda')
    elif args.loss=='dice':
        criterion = DiceLoss().to('cuda')
    elif args.loss=='dicebce':
        criterion = DiceBCELoss().to('cuda')
    
    # Data preproccess (for normalize)
    # 全部data的RGB平均值/255
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]


    # 圖片比例可以透過padding，變成W=H(太佔記憶體空間) -> 改用crop
    # 原圖進來轉tensor和做normalize，train完一次epoch後，
    # 做調整亮度、對比度、旋轉與翻轉等..
    # epoch越多，資料改得越多次
    # ToTensor: range [0, 255] -> [0.0,1.0]
    # Normalize: range [0.0,1.0] -> [-1.0,1.0]
    train_tfms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize([args.input_size_H, args.input_size_W]),
                                    # """ {float}: [max(0, 1 - brightness), 1 + brightness], (tuple): [min, max] """
                                    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                    # transforms.ColorJitter(brightness=(0.0,255.0), contrast=(0.0,255.0), saturation=(0.0,255.0), hue=0.5),
                                    # """
                                    # degrees=360 -> -360° ~ 360° 隨機旋轉
                                    # p=0 -> 不翻轉; p=1 -> 一定翻轉
                                    # """
                                    transforms.RandomRotation(degrees=360),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.Normalize(channel_means, channel_stds)])

    valid_tfms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize([args.input_size_H, args.input_size_W]),
                                    transforms.Normalize(channel_means, channel_stds)])

    mask_tfms = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomRotation(degrees=360),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.Resize([args.input_size_H, args.input_size_W], transforms.InterpolationMode.NEAREST)])


    # DIR_IMG(資料位置)  img_names(資料)  train_tfms(資料transform)
    # DIR_MASK(答案位置) mask_names(答案) mask_tfms(答案transform)
    dataset = ImgDataSet(img_dir=DIR_IMG, img_fnames=img_names, img_transform=train_tfms, mask_dir=DIR_MASK, mask_fnames=mask_names, mask_transform=mask_tfms)
    dataset_valid = ImgDataSet(img_dir=VALID_DIR_IMG, img_fnames=valid_img_names, img_transform=train_tfms, mask_dir=VALID_DIR_MASK, mask_fnames=valid_mask_names, mask_transform=mask_tfms)
    
    if args.sep:
        train_loader = DataLoader(dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
        valid_loader = DataLoader(dataset_valid, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
        print(f'total images = {len(img_names)}')
        print(f'total valid images = {len(valid_img_names)}')
    else:
        # x % training data ; 1-x % validation
        train_size = int(args.split*len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        print(f"\ntrain_dataset is {len(dataset)}")
        print(f"valid_dataset is {len(dataset_valid)}")

        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)

    # for i, (input, target) in enumerate(train_loader):
    #     input_var  = input.cuda()
    #     target_var = target.cuda()
    #     writer.add_image("Training data", input_var[0], i)
    #     writer.add_image("Validation", target_var[0], i)

    model.cuda()

    # train(train_loader, model, criterion, optimizer, validate, args)
    train(train_loader, model, criterion, validate, args, model_output_dir)

    writer.close()