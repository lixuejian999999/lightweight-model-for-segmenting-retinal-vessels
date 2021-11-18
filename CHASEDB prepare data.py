# coding:utf-8

import matplotlib.pyplot as plt  # plt 用于显示图片
import numpy as np
from PIL import Image

import os
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
import cv2

from numpy import *
import h5py
import skimage
from skimage import util

Nimgs = 8
height = 960
width = 999
new_height = 1024
new_width = 1024
channels = 3
patch_h = 64
patch_w = 64
per_img_patch_count = 4500
Max_Length = 10


# 原始图像转换成数据
def get_datasets(imgs_dir, groundTruth_dir, borderMasks_dir=None, train_test="null"):
    imgs = np.empty((Nimgs, height, width, channels))
    groundTruth = np.empty((Nimgs, height, width))
    border_masks=np.empty((Nimgs,height,width))
    for path, subdirs, files in os.walk(imgs_dir):
        for i in range(len(files)):
            # 原始图像转换成数据
            print("original image: " + "Image (" + str(i + 1) + ").jpg")
            img = Image.open(imgs_dir + "/" + "Image (" + str(i + 1) + ").jpg")
            imgs[i] = np.asarray(img)
            # 人工标注转换成数据
            groundTruth_name = "manual (" + str(i + 1) + ").png"
            print("ground truth name: " + groundTruth_name)
            g_Truth = Image.open(groundTruth_dir + "/" + groundTruth_name)
            groundTruth[i] = np.asarray(g_Truth)
            # # 图像边缘转换成数据
            # border_masks_name = str(i+1) + "_test_mask.gif"
            # print("border masks name: " + border_masks_name)
            # b_name = Image.open(borderMasks_dir + "/" + border_masks_name)
            # border_masks[i] = np.asarray(b_name)
    # # 确认数据格式的正确性
    # print("imgs max: " + str(np.max(imgs)))
    # print("imgs min: " + str(np.min(imgs)))
    # assert (np.max(groundTruth) == 255 and np.max(border_masks) == 255)
    # assert (np.min(groundTruth) == 0 and np.min(border_masks) == 0)
    # print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    # reshaping for my standard tensors
    # 这里是将原本imgs的构造，即（Nimgs,height,width，channels）转换成（Nimgs, channels, height, width）这里面的0，1，2，3是指索引
    # 其类似于数学中的坐标，三维数组中这几个数字就分别表示x,y,z
    imgs = np.transpose(imgs, (3, 0, 1, 2))
    assert (imgs.shape == (channels, Nimgs, height, width))
    groundTruth = np.reshape(groundTruth, (Nimgs, 1, height, width))
    # border_masks = np.reshape(border_masks, (Nimgs, 1, height, width))
    assert (groundTruth.shape == (Nimgs, 1, height, width))
    # assert (border_masks.shape == (Nimgs, 1, height, width))
    return imgs, groundTruth


# 将图像转换为成704*704大小
def increasing_width_height_imgs(imgs):
    assert (imgs.shape[1] == 960 and imgs.shape[2] == 999)
    height_img = np.zeros((Nimgs, new_height - 960, imgs.shape[2]))
    imgs = concatenate((imgs, height_img), axis=1)  # 横向连接

    width_img = np.zeros((imgs.shape[0], imgs.shape[1], new_width - 999))
    imgs = concatenate((imgs, width_img), axis=2)  # 纵向连接
    # imgs=imgs[:,:,0:new_width]
    assert (imgs.shape[1] == new_height and imgs.shape[2] == new_width)
    return imgs


# 数据标准化处理
def normlized(original_data):
    imgs_normalized = np.empty(original_data.shape)
    imgs_std = np.std(original_data)
    imgs_mean = np.mean(original_data)
    imgs_normalized = (original_data - imgs_mean) / imgs_std
    for i in range(original_data.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
                np.max(imgs_normalized[i]) - np.min(imgs_normalized[i])))
    assert (np.min(imgs_normalized) == 0 and np.max(imgs_normalized) == 1)
    # print("imgs_normalized_min: ",np.min(imgs_normalized),"imgs_normalized_max: ",np.max(imgs_normalized))
    return imgs_normalized


# 转换成灰度图像
def rgb2gray(imgs):
    r, g, b = imgs[0, :, :, :], imgs[1, :, :, :], imgs[2, :, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    # gray = 0.0 * r + 1.0 * g + 0.0 * b
    return gray


def clahe_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 3D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = clahe.apply(np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i, 0] = cv2.LUT(np.array(imgs[i, 0], dtype=np.uint8), table)
    return new_imgs


# 数据扩展
def data_extend(patch_h, patch_w, per_img_patch_count, original_imgs, original_masks):
    print("数据切片中，每张图片4800，共96000...")
    extended_imgs = np.empty((int(per_img_patch_count * Nimgs / 2), patch_h, patch_w))
    extended_masks = np.empty((int(per_img_patch_count * Nimgs / 2), patch_h, patch_w))
    k = 0
    x, y = 0, 0
    patch_img = []
    patch_mask = []
    original_imgs = np.array(original_imgs)
    for i in range(Nimgs):
        original_img = original_imgs[i]
        original_mask = original_masks[i]
        for m in range(3):
            if m == 1:
                original_img = Image.fromarray(original_img)
                original_mask = Image.fromarray(original_mask)
                original_img = original_img.transpose(Image.ROTATE_90)  # 将图片旋转90度
                original_mask = original_mask.transpose(Image.ROTATE_90)  # 将图片旋转90度
                original_img = np.asarray(original_img)
                original_mask = np.asarray(original_mask)
            elif m == 2:
                original_img = Image.fromarray(original_img)
                original_mask = Image.fromarray(original_mask)
                original_img = original_img.transpose(Image.ROTATE_180)  # 将图片旋转180度
                original_mask = original_mask.transpose(Image.ROTATE_180)  # 将图片旋转180度
                original_img = np.asarray(original_img)
                original_mask = np.asarray(original_mask)
            while (True):
                x = np.random.randint(int(patch_w / 2), new_width - int(patch_w / 2))
                y = np.random.randint(int(patch_h / 2), new_height - int(patch_h / 2))

                patch_img = original_img[y - int(patch_h / 2):y + int(patch_h / 2),
                            x - int(patch_w / 2):x + int(patch_w / 2)]
                patch_mask = original_mask[y - int(patch_h / 2):y + int(patch_h / 2),
                             x - int(patch_w / 2):x + int(patch_w / 2)]
                # 判断当前图像是否98% 都是背景像素
                if np.array(np.where(patch_mask == 0)).shape[1] / (patch_h * patch_w) < 0.9:
                    extended_imgs[k] = patch_img
                    extended_masks[k] = patch_mask
                    k = k + 1
                    if k != 0 and k % float(per_img_patch_count / 3 / 2) == 0:
                        break

    final_extended_imgs = np.empty((int(per_img_patch_count * Nimgs), patch_h, patch_w))
    final_extended_masks = np.empty((int(per_img_patch_count * Nimgs), patch_h, patch_w))
    skimage.util.random_noise(extended_imgs, mode='gaussian', clip=True)
    # 水平反转
    final_extended_imgs[0:int(per_img_patch_count * Nimgs / 2)] = extended_imgs
    final_extended_masks[0:int(per_img_patch_count * Nimgs / 2)] = extended_masks

    final_extended_imgs[int(per_img_patch_count * Nimgs / 2):int(per_img_patch_count * Nimgs)] = cv2.flip(extended_imgs,
                                                                                                          -1)
    final_extended_masks[int(per_img_patch_count * Nimgs / 2):int(per_img_patch_count * Nimgs)] = cv2.flip(
        extended_masks, -1)

    return final_extended_imgs, final_extended_masks


# 测试集数据切分
def divide_test_imgs(imgs, masks):
    new_imgs = np.empty((int(new_height / patch_h) * int(new_height / patch_h) * Nimgs, patch_h, patch_w))
    new_masks = np.empty((int(new_width / patch_w) * int(new_width / patch_w) * Nimgs, patch_h, patch_w))
    itor = 0
    for i in range(Nimgs):
        rows = 0
        cols = 0
        for k in range(int(new_height / patch_h)):
            cols = 0
            for j in range(int(new_width / patch_w)):
                new_imgs[itor] = imgs[i, rows:rows + patch_h, cols:cols + patch_w]
                new_masks[itor] = masks[i, rows:rows + patch_h, cols:cols + patch_w]
                cols = cols + patch_w
                itor = itor + 1
            rows = rows + patch_h
    return new_imgs, new_masks


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


def write_hdf5(arr, fpath):
    with h5py.File(fpath, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)



train_imgs_dir="./drive/My Drive/CHASEDB/training/images"
train_groundTruth_dir="./drive/My Drive/CHASEDB/training/1st_manual"
# train_borderMasks_dir="D:/练习文件/python study/DRIVE/training/mask"
train_imgs,train_groundTruth=get_datasets(train_imgs_dir,train_groundTruth_dir)

test_imgs_dir = "./CHASEDB/test/images"
test_groundTruth_dir = "./CHASEDB/test/1st_manual"
test_border_masks_dir = "./CHASEDB/test/mask"
test_imgs, test_groundTruth= get_datasets(test_imgs_dir, test_groundTruth_dir)
test_border_masks=test_border_masks/255.0
test_imgs[:,8:20,:,:]=train_imgs[:,0:12,:,:]
test_groundTruth[8:20,:,:,:]=train_groundTruth[0:12,:,:,:]

test_imgs = rgb2gray(test_imgs)

test_masks_boder=np.copy(test_imgs)
test_masks_boder=test_masks_boder[0:8,:,:]
test_masks_boder[np.where(test_masks_boder<8)]=0
test_masks_boder[np.where(test_masks_boder>0)]=1
test_masks_boder=test_masks_boder.reshape(8,1,height,width)

test_imgs = increasing_width_height_imgs(test_imgs)
test_imgs = test_imgs.reshape(Nimgs, 1, new_height, new_width)
test_imgs = normlized(test_imgs)
test_imgs = test_imgs * 255.
test_imgs = clahe_equalized(test_imgs)
test_imgs = adjust_gamma(test_imgs, 1.2)
test_imgs = test_imgs / 255.
test_imgs = test_imgs.reshape(Nimgs, new_height, new_width)

test_groundTruth = np.reshape(test_groundTruth, (Nimgs, 960, 999))
test_groundTruth = increasing_width_height_imgs(test_groundTruth)
test_groundTruth = test_groundTruth / 255.

train_imgs=rgb2gray(train_imgs)
train_imgs=increasing_width_height_imgs(train_imgs)
train_imgs=train_imgs.reshape(Nimgs,1,new_height,new_width)
train_imgs=normlized(train_imgs)
train_imgs=train_imgs*255.
train_imgs = clahe_equalized(train_imgs)
train_imgs = adjust_gamma(train_imgs, 1.2)
train_imgs=train_imgs/255.
train_imgs=train_imgs.reshape(Nimgs,new_height,new_width)

train_groundTruth=np.reshape(train_groundTruth,(Nimgs,960,999))
train_groundTruth=increasing_width_height_imgs(train_groundTruth)
train_groundTruth=train_groundTruth/255.


# 数据扩展
Extended_train_imgs,Extended_train_groundTruth=data_extend(patch_h,patch_w,per_img_patch_count,train_imgs,train_groundTruth)
Extended_train_imgs=Extended_train_imgs.reshape(per_img_patch_count*Nimgs,1,patch_h,patch_w)
Extended_train_groundTruth=Extended_train_groundTruth.reshape(per_img_patch_count*Nimgs,1,patch_h,patch_w)
print("训练数据格式为：",Extended_train_imgs.shape,Extended_train_groundTruth.shape)

# 测试数据切分
test_imgs, test_groundTruth = divide_test_imgs(test_imgs, test_groundTruth)

test_imgs=test_imgs[0:256*8,:,:]
test_groundTruth=test_groundTruth[0:256*8,:,:]
# 保存训练数据
print("saving extended train datasets")
assert(Extended_train_imgs.shape==(90000,1,64,64))
print(np.min(Extended_train_imgs),np.max(Extended_train_imgs))
assert(Extended_train_groundTruth.shape==(90000,1,64,64) and np.min(Extended_train_groundTruth)==0 and np.max(Extended_train_groundTruth)==1)
write_hdf5(Extended_train_imgs, "./drive/My Drive/CHASEDB/u_net_data/CHASEDB_dataset_imgs_train.hdf5")
write_hdf5(Extended_train_groundTruth, "./drive/My Drive/CHASEDB/u_net_data/CHASEDB_dataset_groundTruth_train.hdf5")

# 保存测试数据
print("saving test datasets")
test_imgs = test_imgs.reshape(256 * 8, 1, patch_h, patch_w)
test_groundTruth = test_groundTruth.reshape(256 * 8, 1, patch_h, patch_w)
assert (test_imgs.shape == (256 * 8, 1, 64, 64))
print(np.min(test_imgs), np.max(test_imgs))
assert (test_groundTruth.shape == (256 * 8, 1, 64, 64) and np.min(test_groundTruth) == 0 and np.max(
    test_groundTruth) == 1)
write_hdf5(test_imgs, "./CHASEDB/u_net_data/CHASEDB_dataset_imgs_test.hdf5")
write_hdf5(test_groundTruth, "./CHASEDB/u_net_data/CHASEDB_dataset_groundTruth_test.hdf5")
write_hdf5(test_masks_boder, "./CHASEDB/u_net_data/test_border_masks.hdf5")

print("保存完成！")
