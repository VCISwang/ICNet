from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import random
import torchvision.transforms as transforms
import albumentations
import cv2


Blur = albumentations.Blur(blur_limit=7, always_apply=False, p=0.5)
RandomGamma = albumentations.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5)
HueSaturationValue = albumentations.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5)
RGBShift = albumentations.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5)
RandomBrightnessContrast = albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=0.5)
GaussianBlur = albumentations.GaussianBlur(blur_limit=7, always_apply=False, p=0.5)
ChannelShuffle = albumentations.ChannelShuffle(always_apply=False, p=0.5)
InvertImg = albumentations.InvertImg(always_apply=False, p=0.5)
RandomFog = albumentations.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=0.5)
OpticalDistortion = albumentations.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5)
MotionBlur = albumentations.MotionBlur(blur_limit=7, always_apply=False, p=0.5)
MedianBlur = albumentations.MedianBlur(blur_limit=7, always_apply=False, p=0.5)
GaussNoise = albumentations.GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=0.5)
CLAHE = albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5)


def RandomAug(select, img):
    # Convert PIL image to numpy array
    image_np = np.array(img)
    # Apply transformations
    # augmented = self.transform(image=image_np)

    if select == 0:
        image_np = Blur(image=image_np)
    elif select == 1:
        image_np = RandomGamma(image=image_np)
    elif select == 2:
        image_np = HueSaturationValue(image=image_np)
    elif select == 3:
        image_np = RGBShift(image=image_np)
    elif select == 4:
        image_np = RandomBrightnessContrast(image=image_np)
    elif select == 5:
        image_np = GaussianBlur(image=image_np)
    elif select == 6:
        image_np = ChannelShuffle(image=image_np)
    elif select == 7:
        image_np = InvertImg(image=image_np)
    elif select == 8:
        image_np = RandomFog(image=image_np)
    elif select == 9:
        image_np = OpticalDistortion(image=image_np)
    elif select == 10:
        image_np = MotionBlur(image=image_np)
    elif select == 11:
        image_np = MedianBlur(image=image_np)
    elif select == 12:
        image_np = GaussNoise(image=image_np)
    elif select == 13:
        image_np = CLAHE(image=image_np)

    # Convert numpy array to PIL Image
    img = Image.fromarray(image_np['image'])

    return img


class SupDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info('Creating dataset with {} examples'.format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, image, label):
        image = image.convert('RGB')
        label = label.convert('L')

        # color
        color = transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2)
        pil_img_crop = color(image)
        # pil_img_crop = image
        mask_crop = label

        # random flip and rot
        hflip = True
        rot = True
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5
        if hflip:
            pil_img_crop = pil_img_crop.transpose(Image.FLIP_TOP_BOTTOM)
            mask_crop = mask_crop.transpose(Image.FLIP_TOP_BOTTOM)
        if vflip:
            pil_img_crop = pil_img_crop.transpose(Image.FLIP_LEFT_RIGHT)
            mask_crop = mask_crop.transpose(Image.FLIP_LEFT_RIGHT)
        if rot90:
            pil_img_crop = pil_img_crop.transpose(Image.ROTATE_90)
            mask_crop = mask_crop.transpose(Image.ROTATE_90)

        # pool_base = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        # aug_base = random.sample(pool_base, 3)
        # for i in range(3):
        #     select_mode_base = aug_base[i]
        #     pil_img_crop = RandomAug(select_mode_base, pil_img_crop)
            # pil_img_crop = RandomAug(select_mode_base, pil_img_crop)

        # img->numpy
        img_nd = np.array(pil_img_crop)
        mask_nd = np.array(mask_crop)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        if len(mask_nd.shape) == 2:
            mask_nd = np.expand_dims(mask_nd, axis=2)  # expand dims for mask(500, 500)->(500, 500, 1)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))  # (500, 500, 3) -> (3, 500, 500)
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        mask_trans = mask_nd.transpose((2, 0, 1))
        # if mask_trans.max() > 1:
        #     mask_trans = mask_trans / 255

        return img_trans, mask_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        img, mask = self.preprocess(img, mask)

        return {'image': torch.from_numpy(img).float(),
                'mask': torch.from_numpy(mask)}


class SXDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, batch_size):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.lx_batch_size = batch_size

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info('Creating dataset with {} examples'.format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, mask):

        pil_img = pil_img.convert('RGB')
        mask_crop = mask
        mask_crop = mask_crop.convert('L')
        # color
        color = transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2)
        pil_img_crop = color(pil_img)
        # pil_img_crop = pil_img

        # random flip and rot
        hflip = True
        rot = True
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5
        if hflip:
            pil_img_crop = pil_img_crop.transpose(Image.FLIP_TOP_BOTTOM)
            mask_crop = mask_crop.transpose(Image.FLIP_TOP_BOTTOM)
        if vflip:
            pil_img_crop = pil_img_crop.transpose(Image.FLIP_LEFT_RIGHT)
            mask_crop = mask_crop.transpose(Image.FLIP_LEFT_RIGHT)
        if rot90:
            pil_img_crop = pil_img_crop.transpose(Image.ROTATE_90)
            mask_crop = mask_crop.transpose(Image.ROTATE_90)

        # pool_base = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        # aug_base = random.sample(pool_base, 3)
        # for i in range(3):
        #     select_mode_base = aug_base[i]
        #     pil_img_crop = RandomAug(select_mode_base, pil_img_crop)

        # img->numpy
        img_nd = np.array(pil_img_crop)
        mask_nd = np.array(mask_crop)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        if len(mask_nd.shape) == 2:
            mask_nd = np.expand_dims(mask_nd, axis=2)  # expand dims for mask(500, 500)->(500, 500, 1)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))  # (500, 500, 3) -> (3, 500, 500)
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        mask_trans = mask_nd.transpose((2, 0, 1))
        # if mask_trans.max() > 1:
        #     mask_trans = mask_trans / 255

        return img_trans, mask_trans

    def rand_shuffle(self):
        random.shuffle(self.ids)

    def __getitem__(self, i):
        label_number = self.lx_batch_size
        i_num = i * label_number

        for t in range(label_number):
            idx = self.ids[i_num + t]
            # mask_file = self.masks_dir + idx + '*'
            # img_file = self.imgs_dir + idx + '*'
            img_file = glob(self.imgs_dir + idx + '*')
            mask_file = glob(self.masks_dir + idx + '*')
            mask = Image.open(mask_file[0])
            img = Image.open(img_file[0])
            img, mask = self.preprocess(img, mask)
            img = torch.from_numpy(img).unsqueeze(0)
            mask = torch.from_numpy(mask).float().unsqueeze(0)
            if t == 0:
                imgs = img
                masks = mask
            else:
                imgs = torch.cat((imgs, img), 0)
                masks = torch.cat((masks, mask), 0)

        return {'image': imgs, 'mask': masks}  # change 5.25 Byte->Double ?


class ValDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info('Creating dataset with {} examples'.format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, mask_crop):

        pil_img = pil_img.convert('RGB')
        mask_crop = mask_crop.convert('L')

        img_nd = np.array(pil_img)
        mask_nd = np.array(mask_crop)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        if len(mask_nd.shape) == 2:
            mask_nd = np.expand_dims(mask_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        mask_nd = mask_nd.transpose((2, 0, 1))

        return img_trans, mask_nd

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        img, mask = self.preprocess(img, mask)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}


def get_normalized_vector(d):
    d /= 1e-12 + torch.max(torch.abs(d))
    return d / torch.sqrt((1e-6 + torch.sum(torch.pow(d, 2.0))))


def generate_perturbation(x):
    d = torch.normal(torch.zeros(x.size()), torch.ones(x.size()))
    d = get_normalized_vector(d)
    d.requires_grad = False
    return 1 * get_normalized_vector(d)


def generate_perturbation_strong(x):
    d = torch.normal(torch.zeros(x.size()), torch.ones(x.size()))
    d = get_normalized_vector(d)
    d.requires_grad = False
    return 20 * get_normalized_vector(d)


class SUXDataset(Dataset):
    def __init__(self, imgs_dir ):
        self.imgs_dir = imgs_dir

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info('Creating dataset with {} examples'.format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):

        # random flip and rot
        hflip = True
        rot = True
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5
        if hflip:
            pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
        if vflip:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        if rot90:
            pil_img = pil_img.transpose(Image.ROTATE_90)

        # pil_img_1 = pil_img
        # pil_img_2 = pil_img
        #
        # pool = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        # aug = random.sample(pool, 3)
        # for i in range(3):
        #     select_mode = aug[i]
        #     pil_img_1 = RandomAug(select_mode, pil_img_1)

        # color
        # color = transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2)
        # color_strong = transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2)
        color_strong = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        gray = transforms.RandomGrayscale(p=0.2)

        pil_img_1 = color_strong(pil_img)
        pil_img_1 = gray(pil_img_1)
        # pil_img_1 = color(pil_img)
        # pil_img_2 = color(pil_img)
        pil_img_2 = pil_img

        # img->numpy
        img_nd1 = np.array(pil_img_1)
        if len(img_nd1.shape) == 2:
            img_nd1 = np.expand_dims(img_nd1, axis=2)

        img_nd2 = np.array(pil_img_2)
        if len(img_nd2.shape) == 2:
            img_nd2 = np.expand_dims(img_nd2, axis=2)

        # HWC to CHW
        img_trans1 = img_nd1.transpose((2, 0, 1))  # (500, 500, 3) -> (3, 500, 500)
        if img_trans1.max() > 1:
            img_trans1 = img_trans1 / 255

        img_trans2 = img_nd2.transpose((2, 0, 1))  # (500, 500, 3) -> (3, 500, 500)
        if img_trans2.max() > 1:
            img_trans2 = img_trans2 / 255

        img_trans1 = torch.from_numpy(img_trans1).type(torch.FloatTensor)
        img_trans2 = torch.from_numpy(img_trans2).type(torch.FloatTensor)

        r_ulx1 = generate_perturbation_strong(img_trans1).to(torch.float32)
        X_ul1 = img_trans1 + r_ulx1

        # X_ul1 = img_trans1

        X_ul2 = img_trans2

        # save image1 and image2

        # x111 = X_ul1.numpy()
        # x222 = X_ul2.numpy()
        #
        # x111 = x111 * 255
        # x222 = x222 * 255
        #
        # x111 = x111.astype("uint8")
        # x222 = x222.astype("uint8")
        #
        # x111 = x111.transpose((1, 2, 0))
        # x222 = x222.transpose((1, 2, 0))
        #
        # im1 = Image.fromarray(x111)
        # im2 = Image.fromarray(x222)
        # im1.show(title="strong1")
        # im2.show(title="or_image")

        return X_ul1, X_ul2

    def __getitem__(self, i):
        idx = self.ids[i]

        img_file = glob(self.imgs_dir + idx + '*')

        img = Image.open(img_file[0])
        img = img.convert('RGB')

        img1, img2 = self.preprocess(img)

        return {'image1': img1,
                'image2': img2}
