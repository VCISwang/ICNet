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
import random
import shutil
from PIL import Image
import cv2
import pydensecrf.densecrf as crf

from skimage import morphology
from unet import UNet
from LANet import LANet
import deeplab

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import SupDataset
from utils.dataset import SXDataset
from utils.dataset import ValDataset
from utils.dataset import SUXDataset
from torch.utils.data import DataLoader, random_split
from copy import deepcopy


# dir_img = 'data/ISAID/img_75/'
# dir_mask = 'data/ISAID/label_75/'
# ul_img = 'data/ISAID/imgs/'
# val_img = 'data/ISAID/val_imgs/'
# val_mask = 'data/ISAID/val_labels/'

dir_img = 'data/vaihingen/img_100/'
dir_mask = 'data/vaihingen/label_100/'
ul_img = 'data/vaihingen/imgs/'
val_img = 'data/vaihingen/val_imgs/'
val_mask = 'data/vaihingen/val_labels/'


# dir_checkpoint = 'ISAID/results_d/dnet_u_ranpcgn/'
# log_record = '_IS_dnet_u_ranpcgn'

dir_checkpoint = 'vaihingen/results_d/dnet_testt/'
log_record = '_u_dnet_test'


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def label_mask(input):
    pseudo_labels = []

    for i in range(input.shape[0]):
        input_array = np.array(input[i].cpu())
        mask = Image.open('./predict/1.png')
        mask = mask.resize((256, 256))
        mask_np = np.array(mask)

        for x in range(256):
            for y in range(256):
                if input_array[x, y] == 0:
                    mask_np[x, y, 0] = 255
                    mask_np[x, y, 1] = 255
                    mask_np[x, y, 2] = 255
                elif input_array[x, y] == 1:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 0
                    mask_np[x, y, 2] = 255
                elif input_array[x, y] == 2:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 255
                    mask_np[x, y, 2] = 255
                elif input_array[x, y] == 3:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 255
                    mask_np[x, y, 2] = 0
                elif input_array[x, y] == 4:
                    mask_np[x, y, 0] = 255
                    mask_np[x, y, 1] = 255
                    mask_np[x, y, 2] = 0
        pseduo_label_idx = np.expand_dims(mask_np, axis=0)
        pseudo_labels.append(pseduo_label_idx)
    output_image = np.concatenate((pseudo_labels[0], pseudo_labels[1]), axis=0)
    # output_image = Image.fromarray(mask_np.astype(np.int8), mode='RGB')

    return output_image


def pseudo_label(pseu):

    pseudo_labels = []

    for i in range(pseu.shape[0]):
        input_array = np.array(pseu[i].cpu())

        mask = Image.open('./predict/1.png')
        # mask = mask.resize((512, 512))
        mask_np = np.array(mask)

        for x in range(512):
            for y in range(512):
                if input_array[x, y] == 0:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 0
                    mask_np[x, y, 2] = 0
                elif input_array[x, y] == 1:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 0
                    mask_np[x, y, 2] = 127
                elif input_array[x, y] == 2:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 127
                    mask_np[x, y, 2] = 127
                elif input_array[x, y] == 3:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 127
                    mask_np[x, y, 2] = 255
                elif input_array[x, y] == 4:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 63
                    mask_np[x, y, 2] = 63
                elif input_array[x, y] == 5:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 63
                    mask_np[x, y, 2] = 127
                elif input_array[x, y] == 6:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 63
                    mask_np[x, y, 2] = 255
                elif input_array[x, y] == 7:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 127
                    mask_np[x, y, 2] = 191
                elif input_array[x, y] == 8:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 63
                    mask_np[x, y, 2] = 0
                elif input_array[x, y] == 9:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 0
                    mask_np[x, y, 2] = 255
                elif input_array[x, y] == 10:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 100
                    mask_np[x, y, 2] = 155
                elif input_array[x, y] == 11:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 0
                    mask_np[x, y, 2] = 63
                elif input_array[x, y] == 12:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 127
                    mask_np[x, y, 2] = 63
                elif input_array[x, y] == 13:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 191
                    mask_np[x, y, 2] = 127
                elif input_array[x, y] == 14:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 63
                    mask_np[x, y, 2] = 191
                elif input_array[x, y] == 15:
                    mask_np[x, y, 0] = 0
                    mask_np[x, y, 1] = 0
                    mask_np[x, y, 2] = 191
        # mask_np = mask_np.transpose((1, 2, 0))
        pseduo_label_idx = np.expand_dims(mask_np, axis=0)
        pseudo_labels.append(pseduo_label_idx)
    labels = np.concatenate((pseudo_labels[0], pseudo_labels[1]), axis=0)

    return labels


def train_net(net,
              net_ema,
              d_net,
              d_net_ema,
              device,
              epochs=5,
              base_lr=0.01,
              n_classes=16,
              save_cp=True):

    dataset = SXDataset(dir_img, dir_mask, batch_size=2)
    ul_dataset = SUXDataset(ul_img)
    val_set = ValDataset(val_img, val_mask)

    n_train = len(dataset)
    ul_train = len(ul_dataset)
    n_val = len(val_set)

    ul_loader = DataLoader(ul_dataset, batch_size=2, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=log_record)

    global_step = 0
    total_global_step = (ul_train / 2) * epochs

    logging.info('''Starting training:
        Epochs:          {}
        Learning rate:   {}
        Training label_size:   {}
        Validation size: {}
        Checkpoints:     {}
        Device:          {}
    '''.format(epochs, base_lr, n_train, n_val, save_cp, device.type))

    # optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    # m_optimizer = optim.Adam(m_net.parameters(), lr=lr, weight_decay=1e-8)

    optimizer = torch.optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    m_optimizer = torch.optim.SGD(d_net.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)

    if n_classes > 1:
        criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = nn.BCEWithLogitsLoss()

    kl_criterion = nn.KLDivLoss(reduction='none')

    sm = nn.Softmax(dim=1)
    logsm = nn.LogSoftmax(dim=1)

    global iou_best

    iou_best = 0

    for epoch in range(epochs):
        net.train()
        d_net.train()

        vis_t = True

        label_idx = 0
        dataset.rand_shuffle()
        net_evaluator = Evaluator(n_classes)
        net_evaluator_ema = Evaluator(n_classes)

        d_net_evaluator = Evaluator(n_classes)
        d_net_evaluator_ema = Evaluator(n_classes)

        mse_mean = True
        if mse_mean:
            with tqdm(total=ul_train, desc='Epoch {}/{}'.format(epoch + 1, epochs), unit='img') as pbar:
                for ul_batch in ul_loader:
                    batch = dataset.__getitem__(label_idx)
                    imgs = batch['image']
                    true_masks = batch['mask']
                    label_idx = label_idx + 1
                    if label_idx > 49:
                        label_idx = label_idx % 50

                    ul_imgs1 = ul_batch['image1']
                    ul_imgs2 = ul_batch['image2']
                    ul_imgs1 = ul_imgs1.to(device=device, dtype=torch.float32)
                    ul_imgs2 = ul_imgs2.to(device=device, dtype=torch.float32)

                    imgs = imgs.to(device=device, dtype=torch.float32)
                    mask_type = torch.float32 if n_classes == 1 else torch.long
                    true_masks = true_masks.to(device=device, dtype=mask_type)
                    true_masks = torch.squeeze(true_masks)

                    # if global_step % 2 == 0:
                    #     masks_pred = d_net(imgs)
                    # else:
                    #     masks_pred = net(imgs)
                    masks_pred = net(imgs)
                    bce_loss = criterion(masks_pred, true_masks)

                    d_masks_pred = d_net(imgs)
                    d_bce_loss = criterion(d_masks_pred, true_masks)

                    # _, neg_label_sup = torch.min(F.softmax(masks_pred, dim=1), dim=1)
                    # _, d_neg_label_sup = torch.min(F.softmax(d_masks_pred, dim=1), dim=1)
                    # sup_neg_loss = criterion(masks_pred, neg_label_sup).mean() * -0.001
                    # d_sup_neg_loss = criterion(d_masks_pred, d_neg_label_sup).mean() * -0.001

                    # pred_soft = F.softmax(masks_pred, dim=1)
                    # prob_out2 = torch.max(pred_soft, dim=1)
                    # thred_out = torch.lt(prob_out2[0], 0.85)  # < 0.9 comput cross loss
                    # bce_loss = bce_loss * thred_out.float()
                    #
                    # d_pred_soft = F.softmax(d_masks_pred, dim=1)
                    # d_prob_out2 = torch.max(d_pred_soft, dim=1)
                    # thred_out2 = torch.lt(d_prob_out2[0], 0.85)  # < 0.9 comput cross loss
                    # d_bce_loss = d_bce_loss * thred_out2.float()
                    # d_sup_loss = d_bce_loss.mean()

                    sup_loss = bce_loss.mean() + d_bce_loss.mean()  # + sup_neg_loss + d_sup_neg_loss

                    #  unlabel_image
                    # random_x = []
                    # random_y = []
                    # rans_x = int(ul_imgs1.shape[2] / 2)
                    # rans_y = int(ul_imgs1.shape[2] / 2)
                    # cut_num = ul_imgs1.shape[0]
                    # for cn in range(cut_num):
                    #     random_x1 = random.randint(0, rans_x)
                    #     random_x.append(random_x1)
                    #     random_y1 = random.randint(0, rans_y)
                    #     random_y.append(random_y1)

                    N = torch.zeros((ul_imgs1.shape[2], ul_imgs1.shape[3])).cuda()
                    random_x1 = random.randint(0, int(ul_imgs1.shape[2] / 2))
                    random_y1 = random.randint(0, int(ul_imgs1.shape[3] / 2))
                    N[random_x1:(random_x1 + int(ul_imgs1.shape[2] / 2)),
                    random_y1:(random_y1 + int(ul_imgs1.shape[2] / 2))] = 1.0
                    M = torch.abs(N - 1.0)
                    # or_unx = ul_imgs1.clone()
                    # ul_imgs1[0] = or_unx[0] * M + or_unx[1] * N
                    # ul_imgs1[1] = or_unx[1] * M + or_unx[0] * N

                    # random_x = []
                    # random_y = []
                    # rans_x = int(ul_imgs1.shape[2] / 2)
                    # rans_y = int(ul_imgs1.shape[2] / 2)
                    # cut_num = ul_imgs1.shape[0]
                    # for cn in range(cut_num):
                    #     random_x1 = random.randint(0, rans_x)
                    #     random_x.append(random_x1)
                    #     random_y1 = random.randint(0, rans_y)
                    #     random_y.append(random_y1)
                    #
                    # for cn in range(cut_num):
                    #     ul_imgs1[cn, :, random_x[cn]:(random_x[cn] + rans_x),
                    #     random_y[cn]:(random_y[cn] + rans_y)] = imgs[0, :, random_x[cn]:(random_x[cn] + rans_x),
                    #                                             random_y[cn]:(random_y[cn] + rans_y)]
                    num = 1
                    # for cn in range(cut_num):
                    #     ul_imgs1[cn, :, random_x[cn]:(random_x[cn] + rans_x),
                    #     random_y[cn]:(random_y[cn] + rans_y)] = ul_imgs1[num, :, random_x[cn]:(random_x[cn] + rans_x),
                    #                                             random_y[cn]:(random_y[cn] + rans_y)]
                    #     num = num - 1

                    # img_mix = ul_imgs1[0, :].cpu().numpy()
                    # img_mix = img_mix.transpose((1, 2, 0))
                    # img_mix = img_mix * 255
                    # img_mix = img_mix.astype("uint8")
                    # img_mix = Image.fromarray(img_mix)
                    # img_mix.show(title="img_mix")
                    # img_mix.save('./predict/figure/img_mix.png')

                    if global_step % 2 == 0:
                        logit1 = net(ul_imgs1)
                        logit2 = d_net_ema(ul_imgs2)
                    else:
                        logit1 = d_net(ul_imgs1)
                        logit2 = net_ema(ul_imgs2)

                    # sux_pre = torch.argmax(logit1, dim=1)
                    # sux_pre = sux_pre[0, :].cpu().numpy()
                    # sux_pre = Image.fromarray(sux_pre.astype(np.int8), mode='L')
                    # sux_pre.save('./predict/results/' + '2' + '.png')
                    # sux_pre = Image.open('./predict/results/' + '2' + '.png')
                    # sux_pre = label_mask(sux_pre)
                    # sux_pre.show(title='sux_pre')
                    # sux_pre.save('./predict/figure/mix_pre.png')

                    logit_1 = logit1.detach()
                    soft_1 = F.softmax(logit_1, dim=1)
                    pseudo_logit_1, pseudo_label_1 = torch.max(soft_1, dim=1)

                    logit_2 = logit2.detach()
                    soft_2 = (F.softmax(logit_2, dim=1))  # + F.softmax(logit_1, dim=1)) / 2
                    pseudo_logit_2, pseudo_label_2 = torch.max(soft_2, dim=1)

                    # soft_mean = (soft_1 + soft_2) / 2
                    # unc_weight = -1.0 * torch.mean(soft_mean * torch.log(soft_mean + 1e-6), dim=1, keepdim=False)

                    # _, neg_label = torch.min(soft_2, dim=1)

                    del logit2,  # soft_2

                    # uuu = np.array(uncertainty.cpu())

                    # unc_weight = torch.sum(kl_criterion(logsm(logit1), sm(logit_2)), dim=1)
                    # unc_weight = torch.exp(-unc_weight)

                    # uncer1 = (torch.pow(pseudo_logit_1, 2) + torch.pow(pseudo_logit_2, 2)) / 2
                    # uncer2 = torch.pow((pseudo_logit_1 + pseudo_logit_2) / 2, 2)
                    # unc_weight = uncer1 - uncer2

                    # unc_weight = torch.sigmoid(unc_weight)
                    # unc_weight = (unc_weight - torch.mean(unc_weight)) / torch.std(unc_weight)

                    # unc_mean = torch.mean(unc_weight)

                    # unc_w = unc_weight.unsqueeze(dim=1)


                    # for ul_id in range(2):
                    #     tux = ul_imgs2[ul_id, :].cpu().numpy()
                    #     tux = tux.transpose((1, 2, 0))
                    #     tux = tux * 255
                    #     tux = tux.astype("uint8")
                    #     tux = Image.fromarray(tux)
                    #     tux.show(title="tux")
                    # tux.save('./predict/figure/ul_x.png')

                    # for cno in range(cut_num):
                    #     pseudo_label_2[cno, random_x[cno]:(random_x[cno] + rans_x),
                    #     random_y[cno]:(random_y[cno] + rans_y)] = true_masks[0, random_x[cno]:(random_x[cno] + rans_x),
                    #                                                random_y[cno]:(random_y[cno] + rans_y)]

                    nump = 1
                    # for cno in range(cut_num):
                    #     pseudo_label_2[cno, random_x[cno]:(random_x[cno] + rans_x),
                    #     random_y[cno]:(random_y[cno] + rans_y)] = true_masks[0, random_x[cno]:(random_x[cno] + rans_x),
                    #                                               random_y[cno]:(random_y[cno] + rans_y)]
                    #     nump = nump - 1

                    # or_pselabel = pseudo_label_2.clone()
                    # pseudo_label_2[0] = or_pselabel[0] * M.long() + or_pselabel[1] * N.long()
                    # pseudo_label_2[1] = or_pselabel[1] * M.long() + or_pselabel[0] * N.long()

                    if vis_t:
                        label1 = label_mask(pseudo_label_1)
                        label2 = label_mask(pseudo_label_2)
                        writer.add_images('val/un_x1', ul_imgs1.cpu(), epoch)
                        writer.add_images('val/un_x', ul_imgs2.cpu(), epoch)
                        # writer.add_images('val/unc_weight', unc_weight.cpu().unsqueeze(1), global_step=epoch)
                        writer.add_image('val/pseudo_label11', label1[0], global_step=epoch, dataformats='HWC')
                        writer.add_image('val/pseudo_label12', label1[1], global_step=epoch, dataformats='HWC')

                        writer.add_image('val/pseudo_label21', label2[0], global_step=epoch, dataformats='HWC')
                        writer.add_image('val/pseudo_label22', label2[1], global_step=epoch, dataformats='HWC')

                    # for ps_id in range(2):
                    #     mix_pre = pseudo_label[ps_id, :].cpu().numpy()
                    #     mix_pre = Image.fromarray(mix_pre.astype(np.int8), mode='L')
                    #     mix_pre.save('./predict/results/' + '4' + '.png')
                    #     mix_pre = Image.open('./predict/results/' + '4' + '.png')
                    #     mix_pre = label_mask(mix_pre)
                    #     mix_pre.show(title="mix_pre")
                    # mix_pre.save('./predict/figure/mix_label.png')

                    pseudo_loss = criterion(logit1, pseudo_label_2)

                    # neg_pseudo_loss = criterion(logit1, neg_label)

                    # pseudo_w1 = torch.gt(pseudo_logit_2, 0.3).float()
                    # unc_weight = (torch.gt(unc_weight, 0.2).float()) * 0.5 + 1.0
                    # percentage = unc_weight.mean()
                    # final_weight1 = unc_weight.float()
                    # final_weight1 = pseudo_w1.float()
                    # final_weight1 = pseudo_w1.float() * unc_weight.float()

                    # pseudo_loss = pseudo_loss * final_weight1
                    # pseudo_loss = pseudo_loss.sum() / (final_weight1.sum() + 1e-6)
                    pseudo_loss = pseudo_loss.mean()

                    # neg_pseudo_loss = -0.001 * neg_pseudo_loss.mean()

                    # k_pse = sigmoid_rampup(epoch, 50)
                    # pseudo_loss = pseudo_loss_all.mean()
                    loss = sup_loss + pseudo_loss  # + neg_pseudo_loss  # + unc_weight.mean()  # * 0.5

                    if global_step % 100 == 0:
                        writer.add_scalar('train/sup_loss', sup_loss.item(), global_step)
                        writer.add_scalar('train/pseudo_loss', pseudo_loss.item(), global_step)
                        # writer.add_scalar('train/neg_pseudo_loss', neg_pseudo_loss.item(), global_step)

                    pbar.set_postfix(**{'sup_loss ': sup_loss.item(),
                                        'pse_loss': pseudo_loss.item()})
                                        #  'npse_loss': neg_pseudo_loss.item(),
                                         #  'percentage': percentage.item(),

                    current_lr = base_lr * (1 - global_step/total_global_step) ** 0.9
                    writer.add_scalar('train/lr', current_lr, global_step)
                    optimizer.param_groups[0]['lr'] = current_lr
                    m_optimizer.param_groups[0]['lr'] = current_lr

                    optimizer.zero_grad()
                    m_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    nn.utils.clip_grad_value_(d_net.parameters(), 0.1)
                    optimizer.step()
                    m_optimizer.step()

                    alpha = min(1 - 1 / (global_step + 1), 0.999)
                    # buffer_keys = [k for k, _ in net_ema.named_buffers()]
                    # msd = net.state_dict()
                    # esd = net_ema.state_dict()
                    for ema_param, param in zip(net_ema.parameters(), net.parameters()):
                        ema_param.data = ema_param.data * alpha + (1 - alpha) * param.data
                    # for k in buffer_keys:
                    #     esd[k].copy_(msd[k])

                    # buffer_keys_ema = [k for k, _ in d_net_ema.named_buffers()]
                    # msd_ema = d_net.state_dict()
                    # esd_ema = d_net_ema.state_dict()
                    for ema_param, param in zip(d_net_ema.parameters(), d_net.parameters()):
                        ema_param.data = ema_param.data * alpha + (1 - alpha) * param.data
                    # for k in buffer_keys_ema:
                    #     esd_ema[k].copy_(msd_ema[k])

                    net.zero_grad()
                    d_net.zero_grad()

                    pbar.update(ul_imgs1.shape[0])
                    global_step += 1

                    vis_t = False
                    del logit1, pseudo_label_2, masks_pred  # , d_masks_pred

        # val dice + iou +  acc
        val_data = True
        if val_data:
            net.eval()
            d_net.eval()
            net_evaluator.reset()
            net_evaluator_ema.reset()

            net_ema.eval()
            d_net_ema.eval()
            d_net_evaluator.reset()
            d_net_evaluator_ema.reset()

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
                    # seg_predict = remove_small_objects(seg_predict, num_classes=n_classes, threshold=10 * 10)
                    net_evaluator.add_batch(label_val, seg_predict)

                    crop_pre_ema = net_ema(v_imgs)
                    seg_predict_ema = crop_pre_ema.data.cpu().numpy()
                    seg_predict_ema = np.argmax(seg_predict_ema, 1)
                    # seg_predict_ema = remove_small_objects(seg_predict_ema, num_classes=n_classes, threshold=10 * 10)
                    net_evaluator_ema.add_batch(label_val, seg_predict_ema)

                    d_crop_pre = d_net(v_imgs)
                    d_seg_predict = d_crop_pre.data.cpu().numpy()
                    d_seg_predict = np.argmax(d_seg_predict, 1)
                    # d_seg_predict = remove_small_objects(d_seg_predict, num_classes=n_classes, threshold=10 * 10)
                    d_net_evaluator.add_batch(label_val, d_seg_predict)

                    d_crop_pre_ema = d_net_ema(v_imgs)
                    d_seg_predict_ema = d_crop_pre_ema.data.cpu().numpy()
                    d_seg_predict_ema = np.argmax(d_seg_predict_ema, 1)
                    # d_seg_predict_ema = remove_small_objects(d_seg_predict_ema, num_classes=n_classes, threshold=10 * 10)
                    d_net_evaluator_ema.add_batch(label_val, d_seg_predict_ema)

                    pbar.update()

            mIou = net_evaluator.Mean_Intersection_over_Union()
            fscore = net_evaluator.F1score()
            iou = np.nanmean(mIou)
            mF1 = np.nanmean(fscore)

            mIou_ema = net_evaluator_ema.Mean_Intersection_over_Union()
            fscore_ema = net_evaluator_ema.F1score()
            iou_ema = np.nanmean(mIou_ema)
            mF1_ema = np.nanmean(fscore_ema)

            mIou_d = d_net_evaluator.Mean_Intersection_over_Union()
            fscore_d = d_net_evaluator.F1score()
            iou_d = np.nanmean(mIou_d)
            mF1_d = np.nanmean(fscore_d)

            mIou_d_ema = d_net_evaluator_ema.Mean_Intersection_over_Union()
            fscore_d_ema = d_net_evaluator_ema.F1score()
            iou_d_ema = np.nanmean(mIou_d_ema)
            mF1_d_ema = np.nanmean(fscore_d_ema)

            net.train()
            d_net.train()
            net_ema.train()
            d_net_ema.train()

            logging.info('model_val:  ')
            logging.info(' iou: {}, mF1: {}'.format(iou, mF1))
            writer.add_scalar('test/iou', iou, global_step)
            writer.add_scalar('test/mF1', mF1, global_step)

            logging.info('model_val_ema:  ')
            logging.info(' iou_ema: {}, mF1_ema: {}'.format(iou_ema, mF1_ema))
            writer.add_scalar('test/iou_ema', iou_ema, global_step)
            writer.add_scalar('test/mF1_ema', mF1_ema, global_step)

            """ double model val """
            logging.info('d_model_val:')
            logging.info('  iou_d: {}, mF1_d: {}'.format(iou_d, mF1_d))
            writer.add_scalar('test/iou_d', iou_d, global_step)
            writer.add_scalar('test/mF1_d', mF1_d, global_step)

            logging.info('d_model_val_ema:')
            logging.info('  iou_d_ema: {}, mF1_d_ema: {}'.format(iou_d_ema, mF1_d_ema))
            writer.add_scalar('test/iou_d_ema', iou_d_ema, global_step)
            writer.add_scalar('test/mF1_d_ema', mF1_d_ema, global_step)

            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            is_best = iou_d_ema > iou_best
            iou_best = max(iou_d_ema, iou_best)
            if is_best:
                torch.save(d_net_ema.state_dict(), dir_checkpoint + 'best.pth')

            if epoch > 79 and (epoch + 1) % 5 == 0:
                if save_cp:
                    torch.save(net_ema.state_dict(),
                               dir_checkpoint + 'epoch{}.pth'.format(epoch + 1))
                    torch.save(d_net_ema.state_dict(),
                               dir_checkpoint + 'm_epoch{}.pth'.format(epoch + 1))
                    logging.info('Checkpoint {} saved !'.format(epoch + 1))
        del crop_pre, d_crop_pre, v_imgs, v_masks  # , unc_weight
        torch.cuda.empty_cache()

    writer.close()


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




def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.01,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,  # './vaihingen/va_ranp.pth',
                        help='Load model from a .pth file')
    parser.add_argument('-t', '--mean_load', dest='mean_load', type=str, default=False,  # './vaihingen/va_dnet_ranp.pth',
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

        model = UNet(n_channels=3, n_classes=classes, bilinear=True)
        # model = LANet.LANet(in_channels=3, num_classes=classes)
        # model = deeplab.deeplabv3plus_resnet50(num_classes=classes, output_stride=8, pretrained_backbone=True)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    net = create_model()
    net_ema = deepcopy(net)
    for p in net_ema.parameters():
        p.requires_grad_(False)

    d_net = create_model()
    d_net_ema = deepcopy(d_net)
    for p in d_net_ema.parameters():
        p.requires_grad_(False)

    logging.info(parameters_string(net))

    cudnn.benchmark = True

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info('Model loaded from {}'.format(args.load))
        d_net.load_state_dict(
            torch.load(args.mean_load, map_location=device)
        )
        logging.info('mean_Model loaded from {}'.format(args.mean_load))

    net.to(device=device)
    d_net.to(device=device)
    net_ema.to(device=device)
    d_net_ema.to(device=device)

    try:
        train_net(net=net,
                  net_ema=net_ema,
                  d_net=d_net,
                  d_net_ema=d_net_ema,
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
