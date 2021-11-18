# coding:utf-8

import matplotlib.pyplot as plt  # plt 用于显示图片
import numpy as np
import h5py
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sklearn.metrics
from skimage.measure import label
import cv2
import u_net_skeleton
Nimgs=5
height=605
width=700
new_height=704
new_width=704
channels=3
patch_h=64
patch_w=64
per_img_patch_count=6
Max_Length=10

# DRIVE
# Nimgs = 20
# height = 584
# width = 565
# new_height = 640
# new_width = 640
# channels = 3
# patch_h = 64
# patch_w = 64
# per_img_patch_count = 6
# Max_Length = 10


def write_hdf5(arr, fpath):
    with h5py.File(fpath, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


# def connect_imgs(imgs, patch_h, patch_w, height, width):
#   itor = 0
#   new_imgs = np.empty((Nimgs, height, width))
#   for k in range(Nimgs):
#     for i in range(int(height / patch_h)):
#       hang_imgs = imgs[itor]
#       for j in range(int(width / patch_w) - 1):
#         hang_imgs = np.concatenate((hang_imgs, imgs[itor + 1]), axis=1)
#         itor = itor + 1
#       if i == 0:
#         itor = itor + 1
#         lie_imgs = hang_imgs
#       else:
#         itor = itor + 1
#         lie_imgs = np.concatenate((lie_imgs, hang_imgs), axis=0)
#     new_imgs[k] = lie_imgs
#   assert (new_imgs.shape == (Nimgs, height, width))
#   return new_imgs
#
#
# test_groundTruth=load_hdf5("./STARE/u_net_data/STARE_dataset_groundTruth_test.hdf5")
# test_imgs=load_hdf5("./STARE/u_net_data/STARE_dataset_imgs_test.hdf5")
# predictions=np.empty((Nimgs,height,width))
# for i in range(Nimgs):
#   imgs=Image.open("./STARE/STARE-STARE/unet和res2net(64-64)/维度除以4/"+str(i)+"_PIL_precdiction.png")
#   imgs=np.asarray(imgs)
#   imgs=imgs[-height:,0:-width]
#   predictions[i]=np.copy(imgs)
#
#
# test_groundTruth = np.reshape(test_groundTruth, (test_groundTruth.shape[0], patch_h, patch_w))
# test_groundTruth=test_groundTruth[-605:]
# test_groundTruth = connect_imgs(test_groundTruth, patch_h, patch_w, new_height, new_width)
# test_groundTruth = test_groundTruth[:, 0:height, 0:width]
# test_imgs = np.reshape(test_imgs, (test_imgs.shape[0], patch_h, patch_w))
# test_imgs=test_imgs[-605:]
# test_imgs = connect_imgs(test_imgs, patch_h, patch_w, new_height, new_width)
# test_imgs = test_imgs[:, 0:height, 0:width]
# test_imgs=1.0-test_imgs
# write_hdf5(predictions, "./STARE/STARE-STARE/unet和res2net(64-64)/维度除以4/postprocessing/test_predictions.hdf5")
# write_hdf5(test_groundTruth, "./STARE/STARE-STARE/unet和res2net(64-64)/维度除以4/postprocessing/full_test_groundTruth.hdf5")
# write_hdf5(test_imgs, "./STARE/STARE-STARE/unet和res2net(64-64)/维度除以4/postprocessing/full_test_imgs.hdf5")
# width_map=u_net_skeleton.generate_width_map(predictions)
# write_hdf5(width_map, "./STARE/STARE-STARE/unet和res2net(64-64)/维度除以4/postprocessing/width_map.hdf5")

#############################################################################################################
# ############################################################################################################
# test_imgs = load_hdf5("./DRIVE/DRIVE-DRIVE/unet和res2net(64-64)/维度除以4/postprocessing/full_test_imgs.hdf5")
#
#
# test_groundTruth = load_hdf5("./DRIVE/DRIVE-DRIVE/unet和res2net(64-64)/维度除以4/postprocessing/full_test_groundTruth.hdf5")
# predictions = load_hdf5("./DRIVE/DRIVE-DRIVE/unet和res2net(64-64)/维度除以4/postprocessing/test_predictions.hdf5")
# # predictions = predictions / 255.0
# possibility_map = load_hdf5("./DRIVE/DRIVE-DRIVE/unet和res2net(64-64)/维度除以4/postprocessing/possibility_map.hdf5")
# width_map = load_hdf5("./DRIVE/DRIVE-DRIVE/unet和res2net(64-64)/维度除以4/postprocessing/width_map.hdf5")
# test_border_masks=load_hdf5("./DRIVE/u_net_data/test_masks_boder.hdf5")
# test_border_masks = np.reshape(test_border_masks, (Nimgs, height, width))
#
# ############################################################################################################
#############################################################################################################
test_imgs = load_hdf5("./STARE/STARE-STARE/unet和res2net(64-64)/维度除以4/postprocessing/full_test_imgs.hdf5")


test_groundTruth = load_hdf5("./STARE/STARE-STARE/unet和res2net(64-64)/维度除以4/postprocessing/full_test_groundTruth.hdf5")
predictions = load_hdf5("./STARE/STARE-STARE/unet和res2net(64-64)/维度除以4/postprocessing/test_predictions.hdf5")
# predictions = predictions / 255.0
possibility_map = load_hdf5("./STARE/STARE-STARE/unet和res2net(64-64)/维度除以4/postprocessing/possibility_map.hdf5")

width_map = load_hdf5("./STARE/STARE-STARE/unet和res2net(64-64)/维度除以4/postprocessing/width_map.hdf5")
test_border_masks=load_hdf5("./STARE/u_net_data/test_masks_boder.hdf5")
test_border_masks = np.reshape(test_border_masks, (Nimgs, height, width))


def find_endpoint_and_calc_angel(dex_1, imgs):
    dex = [0, 0]
    endpoint = []
    for i, j in zip(dex_1[0], dex_1[1]):
        dex[0] = i
        dex[1] = j
        P = np.empty(8)
        # imgs[dex[0]-1,dex[1]-1]  imgs[dex[0]-1,dex[1]]   imgs[dex[0]-1,dex[1]+1],   0  1  2
        # imgs[dex[0],dex[1]-1]    imgs[dex[0],dex[1]]     imgs[dex[0],dex[1]+1]      7     3
        # imgs[dex[0]+1,dex[1]-1]  imgs[dex[0]+1,dex[1]]   imgs[dex[0]+1,dex[1]+1]    6  5  4
        P[0], P[1], P[2] = imgs[dex[0] - 1, dex[1] - 1], imgs[dex[0] - 1, dex[1]], imgs[dex[0] - 1, dex[1] + 1]
        P[3], P[4], P[5] = imgs[dex[0], dex[1] + 1], imgs[dex[0] + 1, dex[1] + 1], imgs[dex[0] + 1, dex[1]]
        P[6], P[7] = imgs[dex[0] + 1, dex[1] - 1], imgs[dex[0], dex[1] - 1]
        count_1_0 = 0
        P_dex_1 = np.array(np.where(P == 1))
        if P_dex_1.shape[1] <= 1:
            endpoint.append([dex[0], dex[1]])
        elif P_dex_1.shape[1] == 2:
            if np.abs(P_dex_1[0, 0] - P_dex_1[0, 1]) == 1 or np.abs(P_dex_1[0, 0] - P_dex_1[0, 1]) == 7:
                endpoint.append([dex[0], dex[1]])
    return endpoint


def postprocessing(test_imgs, possibility_map, predictions, width_map, distance_threshold=20,   # 20 0.40
                   possibility_threshold=0.4):
    skeleton_prediction = np.copy(predictions)
    dex_1 = np.array(np.where(skeleton_prediction == 1))
    endpoint = find_endpoint_and_calc_angel(dex_1, skeleton_prediction)
    restore_dex = []
    for now_point in endpoint:
        endpoint.remove(now_point)
        for other_point in endpoint:
            distance = np.sqrt(
                np.power((now_point[0] - other_point[0]), 2) + np.power((now_point[1] - other_point[1]), 2))
            if distance <= distance_threshold:
                endpoint.remove(other_point)
                width = (width_map[now_point[0], now_point[1]] + width_map[other_point[0], other_point[1]])/2.0
                if width == 1:  # 厚度为1令其为2，方便后续处理
                    width = 2
                # 横向寻找
                height_1, height_2 = now_point[0] - int(width / 2.0), now_point[0] + int(width / 2.0)
                height_3, height_4 = other_point[0] - int(width / 2.0), other_point[0] + int(width / 2.0)
                wid_1, wid_3 = now_point[1], other_point[1]
                if height_1 <= height_4:
                    if wid_1 <= wid_3:
                        breaken_map = possibility_map[height_1:height_4, wid_1:wid_3]
                        error_dex = np.array(np.where(breaken_map >= 0.5))
                        if error_dex.shape[1] < width*2.0:
                            dex = np.array(np.where(breaken_map >= possibility_threshold))
                            dex = [dex[0] + height_1, dex[1] + wid_1]
                            restore_dex.append(dex)
                    else:
                        breaken_map = possibility_map[height_1:height_4, wid_3:wid_1]
                        error_dex = np.array(np.where(breaken_map >= 0.5))
                        if error_dex.shape[1] < width * 2.0:
                            dex = np.array(np.where(breaken_map >= possibility_threshold))
                            dex = [dex[0] + height_1, dex[1] + wid_3]
                            restore_dex.append(dex)
                else:
                    if wid_1 <= wid_3:
                        breaken_map = possibility_map[height_4:height_1, wid_1:wid_3]
                        error_dex = np.array(np.where(breaken_map >= 0.5))
                        if error_dex.shape[1] < width * 2.0:
                            dex = np.array(np.where(breaken_map >= possibility_threshold))
                            dex = [dex[0] + height_4, dex[1] + wid_1]
                            restore_dex.append(dex)
                    else:
                        breaken_map = possibility_map[height_4:height_1, wid_3:wid_1]
                        error_dex = np.array(np.where(breaken_map >= 0.5))
                        if error_dex.shape[1] < width * 2.0:
                            dex = np.array(np.where(breaken_map >= possibility_threshold))
                            dex = [dex[0] + height_4, dex[1] + wid_3]
                            restore_dex.append(dex)
                # 纵向寻找
                height_1, height_3 = now_point[0], other_point[0]
                wid_1, wid_2 = now_point[1] - int(width / 2.0), now_point[1] + int(width / 2.0)
                wid_3, wid_4 = other_point[1] - int(width / 2.0), other_point[1] + int(width / 2.0)
                if wid_1 <= wid_4:
                    if height_1 <= height_3:
                        breaken_map = possibility_map[height_1:height_3, wid_1:wid_4]
                        error_dex = np.array(np.where(breaken_map >= 0.5))
                        if error_dex.shape[1] < width * 2.0:
                            dex = np.array(np.where(breaken_map >= possibility_threshold))
                            dex = [dex[0] + height_1, dex[1] + wid_1]
                            restore_dex.append(dex)
                    else:
                        breaken_map = possibility_map[height_3:height_1, wid_1:wid_4]
                        error_dex = np.array(np.where(breaken_map >= 0.5))
                        if error_dex.shape[1] < width * 2.0:
                            dex = np.array(np.where(breaken_map >= possibility_threshold))
                            dex = [dex[0] + height_3, dex[1] + wid_1]
                            restore_dex.append(dex)
                else:
                    if height_1 <= height_3:
                        breaken_map = possibility_map[height_1:height_3, wid_4:wid_1]
                        error_dex = np.array(np.where(breaken_map >= 0.5))
                        if error_dex.shape[1] < width * 2.0:
                            dex = np.array(np.where(breaken_map >= possibility_threshold))
                            dex = [dex[0] + height_1, dex[1] + wid_4]
                            restore_dex.append(dex)
                    else:
                        breaken_map = possibility_map[height_3:height_1, wid_4:wid_1]
                        error_dex = np.array(np.where(breaken_map >= 0.5))
                        if error_dex.shape[1] < width * 2.0:
                            dex = np.array(np.where(breaken_map >= possibility_threshold))
                            dex = [dex[0] + height_3, dex[1] + wid_4]
                            restore_dex.append(dex)

    return restore_dex


if __name__ == '__main__':
    error = test_groundTruth - predictions
    fusion_imgs = predictions * test_imgs

    original_predictions = np.copy(predictions)

    final_predictions = np.copy(predictions)
    predictions = predictions * test_border_masks
    import time
    for i in range(Nimgs):
        restore_dex = postprocessing(test_imgs[i], possibility_map[i], predictions[i], width_map[i])
        for dex in restore_dex:
            # final_predictions[i, dex[0], dex[1]] = 2.0
            for x, y in zip(dex[0], dex[1]):
                if final_predictions[i, x, y] != 1.0:
                    final_predictions[i, x, y] = 2.0

    # 转化成RGB格式
    RGB_original_predictions = np.zeros((3, original_predictions.shape[0], original_predictions.shape[1],
                                         original_predictions.shape[2]))
    original_predictions[np.where(original_predictions == 1)] = 2
    original_predictions[np.where(original_predictions == 0)] = 1
    original_predictions[np.where(original_predictions == 2)] = 0
    RGB_original_predictions[2] = original_predictions
    RGB_original_predictions[1] = original_predictions
    RGB_original_predictions[0] = original_predictions

    RGB_label = np.zeros((3, test_groundTruth.shape[0], test_groundTruth.shape[1],
                                         test_groundTruth.shape[2]))
    test_groundTruth[np.where(test_groundTruth == 1)] = 2
    test_groundTruth[np.where(test_groundTruth == 0)] = 1
    test_groundTruth[np.where(test_groundTruth == 2)] = 0
    RGB_label[2] = test_groundTruth
    RGB_label[1] = test_groundTruth
    RGB_label[0] = test_groundTruth
    RGB_label = np.transpose(RGB_label, (1, 2, 3, 0))

    RGB_final_predictions = np.zeros((3, final_predictions.shape[0], final_predictions.shape[1],
                                         final_predictions.shape[2]))
    tempor_final_predictions_1 = np.copy(final_predictions)
    tempor_final_predictions_1[np.where(final_predictions == 0)] = 2
    tempor_final_predictions_1[np.where(final_predictions == 1)] = 0
    tempor_final_predictions_1[np.where(final_predictions == 2)] = 1

    tempor_final_predictions_2 = np.copy(final_predictions)
    tempor_final_predictions_2[np.where(final_predictions == 0)] = 1
    tempor_final_predictions_2[np.where(final_predictions == 2)] = 0
    tempor_final_predictions_2[np.where(tempor_final_predictions_1 == 0)] = 0

    tempor_final_predictions_3 = np.copy(final_predictions)
    tempor_final_predictions_3[np.where(final_predictions == 0)] = 1
    tempor_final_predictions_3[np.where(tempor_final_predictions_1 == 0)] = 0
    RGB_final_predictions[2] = tempor_final_predictions_3
    RGB_final_predictions[1] = tempor_final_predictions_2
    RGB_final_predictions[0] = tempor_final_predictions_1
    RGB_original_predictions = np.transpose(RGB_original_predictions, (1, 2, 3, 0))
    RGB_final_predictions = np.transpose(RGB_final_predictions, (1, 2, 3, 0))

    # 边缘阈值
    final_predictions[np.where(final_predictions == 2.0)] = 1.0

    # 图像显示
    full_imgs = np.concatenate((RGB_label, RGB_original_predictions), axis=2)
    full_imgs = np.concatenate((full_imgs, RGB_final_predictions), axis=2)
    full_imgs = np.uint8(full_imgs * 255)
    for i in range(Nimgs):
        picture = Image.fromarray(full_imgs[i])
        picture.save("./STARE/STARE-STARE/unet和res2net(64-64)/维度除以4/postprocessing2/" + str(i) + ".png")

    # 保存文件
    assert(np.max(final_predictions) == 1.0 and np.min(final_predictions) == 0.0)
    write_hdf5(final_predictions, "./STARE/STARE-STARE/unet和res2net(64-64)/维度除以4/postprocessing2/final_predictions.hdf5")

