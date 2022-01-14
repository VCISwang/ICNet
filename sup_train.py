import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import torch.backends.cudnn as cudnn


from unet import UNet
import deeplab
# from deeplab import modeling

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import SupDataset
from utils.dataset import SXDataset
from utils.dataset import ValDataset
from utils.dataset import SUXDataset
from torch.utils.data import DataLoader, random_split
from LANet import LANet

# dir_img = 'data/ISAID/imgs/'
# dir_mask = 'data/ISAID/labels/'
# val_img = 'data/ISAID/val_imgs/'
# val_mask = 'data/ISAID/val_labels/'

dir_img = 'data/vaihingen/img_100/'
dir_mask = 'data/vaihingen/label_100/'
val_img = 'data/vaihingen/val_imgs/'
val_mask = 'data/vaihingen/val_labels/'


# dir_checkpoint = 'results/75_v3/'
dir_checkpoint = 'vaihingen/results/100sup_LA001ad/'
log_record = '_va100_sup_LA001ad'


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        # Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        # MIoU = np.nanmean(MIoU)
        return MIoU

    def Precision(self):
        precison = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
        return precison

    def Recall(self):
        recall = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        return recall

    def F1score(self):
        precison = self.Precision()
        recall = self.Recall()
        fscore = 2 * precison * recall / (precison + recall)
        return fscore

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def train_net(net,
              device,
              epochs=5,
              base_lr=0.01,
              n_classes=16,
              save_cp=True):

    dataset_sup = SupDataset(dir_img, dir_mask)

    val_set = ValDataset(val_img, val_mask)

    n_train = len(dataset_sup)
    n_val = len(val_set)
    train_loader = DataLoader(dataset_sup, batch_size=2, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    total_global_step = (n_train / 2) * epochs

    writer = SummaryWriter(comment=log_record)
    global_step = 0

    logging.info('''Starting training:
        Epochs:          {}
        Learning rate:   {}
        Training size:   {}
        Validation size: {}
        Checkpoints:     {}
        Device:          {}
    '''.format(epochs, base_lr, n_train, n_val, save_cp, device.type))

    if n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    iou_best = 0
    net_evaluator = Evaluator(n_classes)

    # optimizer = torch.optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=1e-8)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        # optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)

        test = False
        if test:
            net.eval()
            net_evaluator.reset()

            mask_type = torch.float32 if n_classes == 1 else torch.long
            b_val = len(val_loader)  # the number of batch

            with tqdm(total=b_val, desc='Validation round', unit='batch', leave=False) as pbar:
                for v_batch in val_loader:
                    v_imgs, v_masks = v_batch['image'], v_batch['mask']
                    v_imgs = v_imgs.to(device=device, dtype=torch.float32)
                    v_masks = v_masks.to(device=device, dtype=mask_type)
                    v_masks = torch.squeeze(v_masks)
                    label_val = v_masks.cpu().numpy()

                    crop_pre = net(v_imgs)
                    seg_predict = crop_pre.data.cpu().numpy()
                    seg_predict = np.argmax(seg_predict, 1)
                    net_evaluator.add_batch(label_val, seg_predict)

                    pbar.update()

            mIou = net_evaluator.Mean_Intersection_over_Union()
            fscore = net_evaluator.F1score()
            iou = np.nanmean(mIou)
            mF1 = np.nanmean(fscore)

            net.train()

            logging.info('model_val:  ')
            logging.info(' iou: {}, mF1: {}'.format(iou, mF1))

        train = True
        if train:
            with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch + 1, epochs), unit='img') as pbar:
                for batch in train_loader:
                    imgs = batch['image']
                    true_masks = batch['mask']

                    imgs = imgs.to(device=device, dtype=torch.float32)
                    mask_type = torch.float32 if n_classes == 1 else torch.long
                    true_masks = true_masks.to(device=device, dtype=mask_type)
                    true_masks = torch.squeeze(true_masks)

                    masks_pred = net(imgs)
                    loss = criterion(masks_pred, true_masks)

                    epoch_loss += loss.item()
                    writer.add_scalar('train/loss', loss.item(), global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    current_lr = base_lr * (1 - global_step/total_global_step) ** 0.9
                    writer.add_scalar('train/lr', current_lr, global_step)
                    optimizer.param_groups[0]['lr'] = current_lr

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()

                    pbar.update(imgs.shape[0])
                    global_step += 1

        val = True
        if val:
            net.eval()
            net_evaluator.reset()

            mask_type = torch.float32 if n_classes == 1 else torch.long
            b_val = len(val_loader)  # the number of batch

            with tqdm(total=b_val, desc='Validation round', unit='batch', leave=False) as pbar:
                for v_batch in val_loader:
                    v_imgs, v_masks = v_batch['image'], v_batch['mask']
                    v_imgs = v_imgs.to(device=device, dtype=torch.float32)
                    v_masks = v_masks.to(device=device, dtype=mask_type)
                    v_masks = torch.squeeze(v_masks)
                    label_val = v_masks.cpu().numpy()

                    crop_pre = net(v_imgs)
                    seg_predict = crop_pre.data.cpu().numpy()
                    seg_predict = np.argmax(seg_predict, 1)
                    net_evaluator.add_batch(label_val, seg_predict)

                    pbar.update()

            mIou = net_evaluator.Mean_Intersection_over_Union()
            fscore = net_evaluator.F1score()
            iou = np.nanmean(mIou)
            mF1 = np.nanmean(fscore)

            net.train()

            logging.info('model_val:  ')
            logging.info(' iou: {}, mF1: {}'.format(iou, mF1))
            writer.add_scalar('test/iou', iou, global_step)
            writer.add_scalar('test/mF1', mF1, global_step)

            is_best = iou > iou_best
            iou_best = max(iou, iou_best)

            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            if is_best:
                torch.save(net.state_dict(), dir_checkpoint + 'best.pth')

            # scheduler.step(val_score)
            if epoch > 79 and (epoch + 1) % 5 == 0:
                if save_cp:
                    torch.save(net.state_dict(), dir_checkpoint + 'epoch{}.pth'.format(epoch + 1))
                    logging.info('Checkpoint {} saved !'.format(epoch + 1))
            del crop_pre, v_imgs, v_masks
            torch.cuda.empty_cache()

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.01,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,  # './results/v3+_all/epoch95.pth',
                        help='Load model from a .pth file')
    parser.add_argument('-t', '--mean_load', dest='mean_load', type=str, default=False,  # './results/v3+_all/epoch95.pth',  # './test1/epoch260.pth',
                        help='Load mean model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


def parameters_string(module):
    lines = [
        "",
        "List of model parameters:",
        "=========================",
    ]

    row_format = "{name:<40} {shape:>20} ={total_size:>12,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(
            name=name,
            shape=" * ".join(str(p) for p in param.size()),
            total_size=param.numel()
        ))
    lines.append("=" * 75)
    lines.append(row_format.format(
        name="all parameters",
        shape="sum of above",
        total_size=sum(int(param.numel()) for name, param in params)
    ))
    lines.append("")
    return "\n".join(lines)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    torch.cuda.set_device(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {}'.format(device))

    classes = 5

    def create_model(ema=False):

        # model = UNet(n_channels=3, n_classes=classes, bilinear=True)
        model = LANet.LANet(in_channels=3, num_classes=classes)
        # model = deeplab.deeplabv3plus_resnet50(num_classes=classes, output_stride=8, pretrained_backbone=True)

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    net = create_model()
    logging.info(parameters_string(net))

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info('Model loaded from {}'.format(args.load))

    net.to(device=device)

    # faster convolutions, but more memory
    cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  base_lr=args.lr,
                  device=device,
                  n_classes=classes)
    except KeyboardInterrupt:
        # torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
