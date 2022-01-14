from PIL import Image
import random
import os
from os.path import splitext
from os import listdir
import logging
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import cv2

import numpy as np
import os, shutil

import pandas as pd

import seaborn as sns

# from pylab import *

# rcParams['axes.unicode_minus'] = True
#
# mpl.rcParams['font.sans-serif'] = ['KaiTi']
# mpl.rcParams['font.serif'] = ['simkai']

# import seaborn as sns
# sns.set_style("darkgrid", {"font.sans-serif": ['KaiTi', 'Arial']})

crop = False
if crop:
    imgs_dir = '../data/ul_imgs'
    # masks_dir = '../data/masks'

    ids = [splitext(file)[0] for file in listdir(imgs_dir)
           if not file.startswith('.')]
    logging.info('Creating dataset with {} examples'.format(len(ids)))

    for i in range(len(ids)):
        idx = ids[i]
        # mask_file = str(masks_dir + '/' + idx + '.tif')
        img_file = str(imgs_dir + '/' + idx + '.tif')
        # mask = Image.open(mask_file)
        img = Image.open(img_file)
        for x in range(10):
            for y in range(10):
                box = (500 * x, 500 * y, 500 * (x + 1), 500 * (y + 1))
                region_img = img.crop(box)
                # region_mask = mask.crop(box)
                # region_img.show()
                # region_mask.show()
                region_img.save('../data/ul_img_aug/{}_{}{}.tif'.format(idx, x, y))
                # region_mask.save('../data/mask_aug/{}_{}{}.tif'.format(idx, x, y))

resize = False
if resize:

    imgs_dir = '../data/ISAID/imgs'
    masks_dir = '../data/ISAID/masks'

    ids = [splitext(file)[0] for file in listdir(imgs_dir)
           if not file.startswith('.')]
    logging.info('Creating dataset with {} examples'.format(len(ids)))

    random.shuffle(ids)

    for i in range(len(ids)):
        idx = ids[i]
        mask_file = str(masks_dir + '/' + idx + '_instance_color_RGB.png')
        img_file = str(imgs_dir + '/' + idx + '.png')
        mask = Image.open(mask_file)
        img = Image.open(img_file)

        # region_img = img.crop(box)
        region_img = img.resize((512, 512))

        # region_mask = mask.crop(box)
        region_mask = mask.resize((512, 512))
        # region_img.show()
        # region_mask.show()
        if i < 1200:
            region_img.save('../data/imgs/{}.png'.format(idx))
            region_mask.save('../data/masks/{}.png'.format(idx))
        else:
            region_img.save('../data/val_imgs/{}.png'.format(idx))
            region_mask.save('../data/val_masks/{}.png'.format(idx))


val_resize = False
if val_resize:

    imgs_dir = '../data/val_imgs'
    masks_dir = '../data/val_label'

    ids = [splitext(file)[0] for file in listdir(imgs_dir)
           if not file.startswith('.')]
    logging.info('Creating dataset with {} examples'.format(len(ids)))

    for i in range(len(ids)):
        idx = ids[i]
        mask_file = str(masks_dir + '/' + idx + '.tif')
        img_file = str(imgs_dir + '/' + idx + '.jpg')
        mask = Image.open(mask_file)
        img = Image.open(img_file)

        # region_img = img.crop(box)
        region_img = img.resize((968, 608))

        # region_mask = mask.crop(box)
        region_mask = mask.resize((968, 608))
        # region_img.show()
        # region_mask.show()
        region_img.save('../data/scale/val_imgs/{}.jpg'.format(idx))
        region_mask.save('../data/scale/val_label/{}.tif'.format(idx))


def IsSubString(SubStrList,Str):
    '''
    #判断字符串Str是否包含序列SubStrList中的每一个子字符串
    #>>>SubStrList=['F','EMS','txt']
    #>>>Str='F06925EMS91.txt'
    #>>>IsSubString(SubStrList,Str)#return True (or False)
    '''
    flag=True
    for substr in SubStrList:
        if not(substr in Str):
            flag=False

    return flag


select = False
n = 0
if select:
    path = '/home/chrisd/wjx/DOTA/data/ISAID/imgs/'
    FlagStr = '.png'
    imgNames = os.listdir(path)
    random.shuffle(imgNames)
    for img in imgNames:
        if (IsSubString(FlagStr, img)):
            n = n + 1
            fullfilename = str(img).split('.')
            region_img = Image.open(path + img)
            region_mask = Image.open('/home/chrisd/wjx/DOTA/data/ISAID' + '/labels/' + fullfilename[0] + '_instance_color_RGB.png')
            # region_img = images_i.resize((512, 512))
            # region_mask = mask_i.resize((512, 512))
            if n < 1201:
                region_img.save('../data/819/imgs/' + fullfilename[0] + '.png')
                region_mask.save('../data/819/labels/' + fullfilename[0] + '.png')
            else:
                region_img.save('../data/819/val_imgs/' + fullfilename[0] + '.png')
                region_mask.save('../data/819/val_labels/' + fullfilename[0] + '.png')
            # images_i.save('./data/78/imgs/' + fn + '_' + img)
            # new_json = './data/78/json/' + fn + '_' + img
            # os.rename(fullfilename, new_json)


mask_label = False
if mask_label:
    masks_dir = '/home/chrisd/wjx/DOTA/data/ISAID/masks'

    ids = [splitext(file)[0] for file in listdir(masks_dir)
           if not file.startswith('.')]

    for i in range(len(ids)):
        idx = ids[i]
        mask_file = str(masks_dir + '/' + idx + '.png')

        label_file = Path('/home/chrisd/wjx/DOTA/data/ISAID/labels/' + idx + '.png')

        if label_file.is_file():
            print(str(idx) + 'already process')
        else:
            mask = Image.open(mask_file)
            mask_np = np.array(mask)

            label = np.ones([mask_np.shape[0], mask_np.shape[1]], dtype=int)

            for x in range(mask_np.shape[0]):
                for y in range(mask_np.shape[1]):
                    rgb = mask_np[x, y, :]
                    if rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 0:
                        label[x, y] = 0
                    elif rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 127:
                        label[x, y] = 1
                    elif rgb[0] == 0 and rgb[1] == 127 and rgb[2] == 127:
                        label[x, y] = 2
                    elif rgb[0] == 0 and rgb[1] == 127 and rgb[2] == 255:
                        label[x, y] = 3
                    elif rgb[0] == 0 and rgb[1] == 63 and rgb[2] == 63:
                        label[x, y] = 4
                    elif rgb[0] == 0 and rgb[1] == 63 and rgb[2] == 127:
                        label[x, y] = 5
                    elif rgb[0] == 0 and rgb[1] == 63 and rgb[2] == 255:
                        label[x, y] = 6
                    elif rgb[0] == 0 and rgb[1] == 127 and rgb[2] == 191:
                        label[x, y] = 7
                    elif rgb[0] == 0 and rgb[1] == 63 and rgb[2] == 0:
                        label[x, y] = 8
                    elif rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 255:
                        label[x, y] = 9
                    elif rgb[0] == 0 and rgb[1] == 100 and rgb[2] == 155:
                        label[x, y] = 10
                    elif rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 63:
                        label[x, y] = 11
                    elif rgb[0] == 0 and rgb[1] == 127 and rgb[2] == 63:
                        label[x, y] = 12
                    elif rgb[0] == 0 and rgb[1] == 191 and rgb[2] == 127:
                        label[x, y] = 13
                    elif rgb[0] == 0 and rgb[1] == 63 and rgb[2] == 191:
                        label[x, y] = 14
                    elif rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 191:
                        label[x, y] = 15
            out_label = Image.fromarray(label.astype(np.int8), mode='L')
            out_label.save('/home/chrisd/wjx/DOTA/data/ISAID/labels/' + idx + '.png')
            print(str(idx) + 'now process')

move_test = False
if move_test:
    def mymovefile(srcfile, dstpath):  # 移动函数
        if not os.path.isfile(srcfile):
            print("%s not exist!" % (srcfile))
        else:
            fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
            if not os.path.exists(dstpath):
                os.makedirs(dstpath)  # 创建路径
            shutil.move(srcfile, dstpath + fname)  # 移动文件
            print("move %s -> %s" % (srcfile, dstpath + fname))


    src_dir = '../data/819/1/val_imgs/'
    dst_dir = '../data/819/1/test_imgs/'

    label_src = '../data/819/1/val_labels/'
    label_dst = '../data/819/1/test_labels/'
    # 目的路径记得加斜杠
    # src_file_list = glob(src_dir + '*')  # glob获得路径下所有文件，可根据需要修改

    ids = [splitext(file)[0] for file in listdir(src_dir)
           if not file.startswith('.')]
    logging.info('Creating dataset with {} examples'.format(len(ids)))

    random.shuffle(ids)
    t = 0
    for i in range(len(ids)):
        idx = ids[i]
        t = t + 1
        if t < 1333:
            mymovefile(src_dir + idx + '.png', dst_dir)
            mymovefile(label_src + idx + '.png', label_dst)


select_semi = True
if select_semi:

    imgs_dir = '../data/Aeroscapes/imgs/'
    masks_dir = '../data/Aeroscapes/labels/'

    ids = [splitext(file)[0] for file in listdir(imgs_dir)
           if not file.startswith('.')]
    logging.info('Creating dataset with {} examples'.format(len(ids)))

    random.shuffle(ids)

    for i in range(len(ids)):
        idx = ids[i]
        mask_file = glob(masks_dir + idx + '*')
        img_file = glob(imgs_dir + idx + '*')

        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        if i < len(ids) * 1/16:
            img.save('../data/Aeroscapes/img_150/{}.png'.format(idx))
            mask.save('../data/Aeroscapes/label_150/{}.png'.format(idx))
        if i < len(ids) * 1/8:
            img.save('../data/Aeroscapes/img_300/{}.png'.format(idx))
            mask.save('../data/Aeroscapes/label_300/{}.png'.format(idx))
        if i < len(ids) * 1/4:
            img.save('../data/Aeroscapes/img_600/{}.png'.format(idx))
            mask.save('../data/Aeroscapes/label_600/{}.png'.format(idx))


creat_val = False
if creat_val:

    imgs_dir = '../data/819/val_imgs'
    masks_dir = '../data/819/val_labels'

    ids = [splitext(file)[0] for file in listdir(imgs_dir)
           if not file.startswith('.')]
    logging.info('Creating dataset with {} examples'.format(len(ids)))

    random.shuffle(ids)

    size = 512
    g_count = 1
    for k in range(len(ids)):

        # count = 0
        idx = ids[k]
        image = cv2.imread(imgs_dir + '/' + idx + '.png')
        label = cv2.imread(masks_dir + '/' + idx + '.png', 0)
        X_height, X_width = image.shape[0], image.shape[1]
        # while count < image_each:
        #     random_width = random.randint(0, X_width - size - 1)
        #     random_height = random.randint(0, X_height - size - 1)
        #     image_ogi = image[random_height: random_height + size, random_width: random_width + size, :]
        #     label_ogi = label[random_height: random_height + size, random_width: random_width + size]
        #     if count < 50:
        #         image_d, label_d = image_ogi, label_ogi
        #     else:
        #         image_d, label_d = data_augment(image_ogi, label_ogi)
        #
        #     image_path.append(train_image_path+'%05d.png' % g_count)
        #     label_path.append(train_label_path+'%05d.png' % g_count)
        #     cv2.imwrite((train_image_path+'%05d.png' % g_count), image_d)
        #     cv2.imwrite((train_label_path+'%05d.png' % g_count), label_d)
        #
        #     count += 1
        #     g_count += 1
        i = 0
        while (i + 1) * size <= X_height:  # i行
            j = 0
            while (j + 1) * size <= X_width:  # j列，假设图片是矩形的
                image_d = image[i * size:(i + 1) * size, j * size:(j + 1) * size]
                label_d = label[i * size:(i + 1) * size, j * size:(j + 1) * size]
                j += 1
                # image_path.append(train_image_path + 'slide' + '%05d.png' % g_count)
                # label_path.append(train_label_path+'%05d.png' % g_count)

                np_label = np.array(label_d)
                sum_label = np_label.sum()
                print(sum_label)
                if sum_label > 1000:
                    cv2.imwrite(('../data/819/1/val_imgs/' + str(k) + 's' + '%05d.png' % g_count), image_d)
                    cv2.imwrite(('../data/819/1/val_labels/' + str(k) + 's' + '%05d.png' % g_count), label_d)
                # save_path = out_path + str(k) + '_' + str(i) + '_' + str(j) + '.jpg'
                # cv.imwrite(save_path, part)

                    g_count += 1
            i += 1


select_ulimg = False
if select_ulimg:

    imgs_dir_all = '../data/ul_900'
    imgs_dir_semi = '../data/img_300'

    ids = [splitext(file)[0] for file in listdir(imgs_dir_semi)
           if not file.startswith('.')]
    logging.info('Creating dataset with {} examples'.format(len(ids)))

    random.shuffle(ids)

    for i in range(len(ids)):
        idx = ids[i]
        img_file = str(imgs_dir_all + '/' + idx + '.png')

        os.remove(img_file)


mat = False
if mat:
    # x = (15, 40, 80, 155)
    # y1 = (72.30, 75.31, 76.19, 78.35)
    # y2 = (73.83, 76.08, 76.82, 78.61)
    # y3 = (74.40, 76.64, 77.12, 78.61)
    #
    # plt.xlabel("The number of labeled sample", fontsize=16)
    # plt.ylabel("MIou", fontsize=16)
    #
    # plt.plot(x, y1, color='r', marker='o', mec='r', label='baseline')
    # plt.plot(x, y2, color='g', marker='*', mec='g', label='CR')
    # plt.plot(x, y3, color='b', marker='x', mec='b', label='CR + AUP')
    #
    # plt.legend(loc="lower right")
    # plt.axis([0, 160, 60, 80])
    # plt.title('Inria Aerial Image', fontsize=24)
    # plt.show()

    # x = (400, 1350, 2700, 5400)
    # y1 = (51.50, 56.21, 59.11, 62.41)
    # y2 = (52.54, 57.09, 59.62, 62.63)
    # y3 = (53.63, 58.01, 60.22, 62.63)
    #
    # plt.xlabel("The number of labeled sample", fontsize=16)
    # plt.ylabel("MIou", fontsize=16)
    #
    # plt.plot(x, y1, color='r', marker='o', mec='r', label='baseline')
    # plt.plot(x, y2, color='g', marker='*', mec='g', label='CR')
    # plt.plot(x, y3, color='b', marker='x', mec='b', label='CR + AUP')
    #
    # plt.legend(loc="lower right")
    # plt.axis([0, 5400, 50, 70])
    # plt.title('Road Extraction', fontsize=24)
    # plt.show()

    # x = (100, 300, 600, 1200)
    # y1 = (36.76, 41.46, 43.92, 47.45)
    # y2 = (39.31, 42.56, 44.86, 47.73)
    # y3 = (39.85, 42.88, 45.02, 47.73)
    #
    # plt.xlabel("The number of labeled sample", fontsize=16)
    # plt.ylabel("MIou", fontsize=16)
    #
    # plt.plot(x, y1, color='r', marker='o', mec='r', label='baseline')
    # plt.plot(x, y2, color='g', marker='*', mec='g', label='CR')
    # plt.plot(x, y3, color='b', marker='x', mec='b', label='CR + AUP')
    #
    # plt.legend(loc="lower right")
    # plt.axis([0, 1200, 30, 50])
    # plt.title('ISAID', fontsize=24)
    # plt.show()

    # x = (0, 0.1, 0.2, 0.3, 0.5)
    # y1 = (67.25, 67.32, 67.52, 67.43, 67.10)
    # y2 = (71.04, 71.22, 71.50, 71.52, 71.11)
    # # y3 = (53.63, 58.01, 60.22, 62.63)
    #
    # plt.xlabel("Color Jitter range", fontsize=16)
    # plt.ylabel("MIou", fontsize=16)
    #
    # plt.plot(x, y1, color='r', marker='o', mec='r', label='15 label')
    # plt.plot(x, y2, color='g', marker='*', mec='g', label='40 label')
    # # plt.plot(x, y3, color='b', marker='x', mec='b', label='CR + AUP')
    #
    # plt.legend(loc="lower right")
    # plt.axis([0, 0.6, 65, 75])
    # plt.title('Aerial Dataset', fontsize=24)
    # plt.show()

    x = (100, 300, 600, 1200)
    y1 = (30.02, 31.48, 32.74, 37.10)
    y2 = (31.14, 32.22, 33.35, 36.92)
    y3 = (33.05, 33.83, 34.52, 37.28)
    y4 = (33.51, 34.26, 35.06, 37.37)
    # y3 = (53.63, 58.01, 60.22, 62.63)

    plt.xlabel("labeled images", fontsize=11)
    plt.ylabel("mIoU", fontsize=11)

    plt.plot(x, y4, color='r', marker='*', mec='r', linewidth=2,  linestyle='-', label='paste 1/4')

    plt.plot(x, y3, color='b', marker='x', mec='b', linewidth=1.5, linestyle='-.', label='paste 1/2')

    plt.plot(x, y2, color='g', marker='o', mec='g', linewidth=1.5, linestyle='--', label='random size')

    plt.plot(x, y1, color='goldenrod', marker='+', linewidth=2, linestyle=':', label='no paste')
    # plt.plot(x, y3, color='b', marker='x', mec='b', label='CR + AUP')

    plt.legend(loc="lower right")
    # plt.axis([100, 1200, 28, 38])

    plt.xlim((100, 1200))
    plt.ylim((28.00, 38.00))
    plt.xticks(np.arange(100, 1200, 100))
    plt.yticks(np.arange(28.00, 38.00, 1))
    # plt.title('iSAID Dataset', fontsize=24)
    plt.show()

    a = 0

mat = False
if mat:
    # A = [-70.97, -67.01, -69.31, -42.61, -30.16]
    # B = [-17.40, -31.21, -25.41, -1.16, 35.53]
    # C = [52.15, 82.49, 94.88, 88.20, 57.73]
    # D = [96.32, 87.48, 60.59, 47.28, 25.76]
    #
    # my_font = fm.FontProperties(fname=r'/usr/share/fonts/winFonts/simkai.ttf')
    #
    # plt.figure(figsize=(6, 6), dpi=60)  # 设置画板
    # box = plt.boxplot([A, B, C, D], labels=[' ', ' ', ' ', ' '],
    #             sym=' ',  # 异常点的形状，参照marker的形状
    #             vert=True,  # 图是否竖着画
    #             whis=1.5,  # 上下须与上下四分位的距离，默认为1.5倍的四分位差
    #             patch_artist=True,
    #             showfliers=True,
    #             showmeans=False)  # 是否显示异常值
    #
    # colors = ['lightblue', 'lightgreen', 'lightpink', 'lightgray']
    # for patch, color in zip(box['boxes'], colors):
    #     patch.set_color(color)
    #
    # plt.xlabel(" 重庆     武汉     沈阳     苏州 ", fontproperties=my_font, fontsize=18)
    #
    # plt.ylabel('人口性别结构指数', fontproperties=my_font, fontsize=18)
    # plt.savefig('boxplot.pdf')
    # plt.show()

    # A = [108.94, 108.51, 107.19, 106.18, 105.11]
    # B = [105.82, 106.62, 105.60, 104.43, 103.53]
    # C = [101.97, 100.63, 99.23, 97.95, 96.76]
    # D = [98.83, 97.84, 97.27, 96.38, 95.38]
    #
    # my_font = fm.FontProperties(fname=r'/usr/share/fonts/winFonts/simkai.ttf')
    # plt.figure(figsize=(6, 6), dpi=60)  # 设置画板
    # box = plt.boxplot([A, B, C, D], labels=[' ', ' ', ' ', ' '],
    #             sym=' ',  # 异常点的形状，参照marker的形状
    #             vert=True,  # 图是否竖着画
    #             whis=1.5,  # 四分位的距离，默认为1.5倍的四分位差
    #             patch_artist=True,
    #             showfliers=True,
    #             showmeans=False)  # 是否显示异常值
    #
    # colors = ['lightblue', 'lightgreen', 'lightpink', 'lightgray']
    # for patch, color in zip(box['boxes'], colors):
    #     patch.set_color(color)
    #
    # plt.xlabel(" 重庆      武汉     沈阳     苏州 ", fontproperties=my_font, fontsize=18)
    #
    # plt.ylabel('人口性别比', fontproperties=my_font, fontsize=18)
    # plt.savefig('boxplot.pdf')
    # plt.show()

    # 密度图

    # Import Data
    # df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
    x = (2000, 2005, 2010, 2015, 2019)

    A = [43.89, 45.30, 49.68, 53.08, 53.90]
    B = [44.15, 46.02, 50.36, 52.15, 52.17]
    C = [31.37, 32.57, 35.48, 37.85, 41.32]
    # data = {"A"[43.89, 45.30, 49.68, 53.08, 53.90],
    # "B"[44.15, 46.02, 50.36, 52.15, 52.17], "C"[31.37, 32.57, 35.48, 37.85, 41.32]}
    # Draw Plot
    plt.figure(figsize=(8, 6), dpi=60)
    plt.plot(x, A, color='r', marker='*', mec='r', linewidth=2,  linestyle='-', label='ben')

    plt.plot(x, B, color='b', marker='x', mec='b', linewidth=1.5, linestyle='-.', label='shuo')

    plt.plot(x, C, color='g', marker='o', mec='g', linewidth=1.5, linestyle='--', label='bo')

    plt.legend(loc="lower right")
    # plt.axis([100, 1200, 28, 38])

    # plt.xlim((1995, 2020))
    plt.ylim((25.00, 60.00))
    # plt.xticks(np.arange(2000, 2020, 5))
    plt.yticks(np.arange(25.00, 60.00, 5))

    # plt.figure(figsize=(8, 6), dpi=60)
    # sns.kdeplot(A, shade=True, color="g", label="ben", alpha=.7)
    # sns.kdeplot(B, shade=True, color="black", label="shuo", alpha=.7)
    # sns.kdeplot(C, shade=True, color="dodgerblue", label="bo", alpha=.7)

    # sns.kdeplot(df.loc[df['cyl'] == 8, "cty"], shade=True, color="orange", label="Cyl=8", alpha=.7)

    # plt.xlabel(" 重庆      武汉     沈阳     苏州 ", fontproperties=my_font, fontsize=18)
    # Decoration
    plt.title('Density Plot of City Mileage by n_Cylinders', fontsize=18)
    plt.legend()
    plt.show()

    a = 0