# coding:utf-8

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sklearn.metrics
from skimage.measure import label
# import skeleton_width
from tensorflow.keras.models import model_from_json
from tqdm import tqdm
import time
Nimgs = 20
height = 584
width = 565
new_height = 640
new_width = 640
channels = 3
patch_h = 64
patch_w = 64
per_img_patch_count = 4500
Max_Length = 10


# Nimgs=5
# height=605
# width=700
# new_height=704
# new_width=704
# channels=3
# patch_h=64
# patch_w=64
# per_img_patch_count=4500
# Max_Length=10


def write_hdf5(arr, fpath):
    with h5py.File(fpath, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape) == 3)  # 3D array: (Npatches,height*width,2)
    # assert (pred.shape[2]==2 )  #check the classes are 2
    pred_images = np.empty((pred.shape[0], pred.shape[1]))  # (Npatches,height*width)
    if mode == "original":
        if mode == "original":
            for i in range(pred.shape[0]):
                for pix in range(pred.shape[1]):
                    pred_images[i, pix] = pred[i, pix, 1]
    elif mode == "Pixel":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i, pix, 1] >= 0.5:
                    pred_images[i, pix] = 1
                else:
                    pred_images[i, pix] = 0
    elif mode == "Joint":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i, pix, 0] >= 0.5:
                    pred_images[i, pix] = pred[i, pix, 1]
                elif pred[i, pix, 1] >= 0.5:
                    pred_images[i, pix] = pred[i, pix, 1]
                elif pred[i, pix, 2] >= 0.5:
                    pred_images[i, pix] = pred[i, pix, 1]

    else:
        print("mode " + str(mode) + " not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images, (pred_images.shape[0], patch_height, patch_width))
    return pred_images


def connect_imgs(imgs, patch_h, patch_w, height, width):
    itor = 0
    new_imgs = np.empty((Nimgs, height, width))
    for k in range(Nimgs):
        for i in range(int(height / patch_h)):
            hang_imgs = imgs[itor]
            for j in range(int(width / patch_w) - 1):
                hang_imgs = np.concatenate((hang_imgs, imgs[itor + 1]), axis=1)
                itor = itor + 1
            if i == 0:
                itor = itor + 1
                lie_imgs = hang_imgs
            else:
                itor = itor + 1
                lie_imgs = np.concatenate((lie_imgs, hang_imgs), axis=0)
        new_imgs[k] = lie_imgs
    assert (new_imgs.shape == (Nimgs, height, width))
    return new_imgs


# 计算TP,TN,FP,FN
def calc_Evaluation(pre_labels, actual_labels, test_border_masks):
    pre_labels = pre_labels[:, 0:height, 0:width]
    actual_labels = actual_labels[:, 0:height, 0:width]
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(pre_labels.shape[0]):
        for j in range(pre_labels.shape[1]):
            for k in range(pre_labels.shape[2]):
                if test_border_masks[i, 0, j, k] == 1:
                    if pre_labels[i, j, k] == 1 and actual_labels[i, j, k] == 1:
                        TP = TP + 1
                    elif pre_labels[i, j, k] == 0 and actual_labels[i, j, k] == 0:
                        TN = TN + 1
                    elif pre_labels[i, j, k] == 1 and actual_labels[i, j, k] == 0:
                        FP = FP + 1
                    elif pre_labels[i, j, k] == 0 and actual_labels[i, j, k] == 1:
                        FN = FN + 1
    Se = float(TP) / (TP + FN)
    Sp = float(TN) / (TN + FP)
    Acc = float(TP + TN) / (TN + TP + FN + FP)
    Pr = float(TP) / (TP + FP)
    Pixel_F1_Score = 2.0 * Pr * Se / (Pr + Se)

    dice = 2 * TP / (2 * TP + FP + FN)
    return Se, Sp, Acc, Pixel_F1_Score, dice


# 计算TP,TN,FP,FN
def calc_mean_dice(pre, actual_labels, test_border_masks):
    pre = pre[:, 0:height, 0:width]
    actual_labels = actual_labels[:, 0:height, 0:width]
    TP, TN, FP, FN = 0, 0, 0, 0
    thres = list(range(0, 100, 5))
    Se, Sp, Acc, dice = [], [], [], []
    for t in tqdm(thres):
        t = t / 100.
        pre_labels = np.copy(pre)
        pre_labels[pre_labels >= t] = 1.0
        pre_labels[pre_labels < t] = 0.0
        for i in range(pre_labels.shape[0]):
            for j in range(pre_labels.shape[1]):
                for k in range(pre_labels.shape[2]):
                    if test_border_masks[i, 0, j, k] == 1:
                        if pre_labels[i, j, k] == 1 and actual_labels[i, j, k] == 1:
                            TP = TP + 1
                        elif pre_labels[i, j, k] == 0 and actual_labels[i, j, k] == 0:
                            TN = TN + 1
                        elif pre_labels[i, j, k] == 1 and actual_labels[i, j, k] == 0:
                            FP = FP + 1
                        elif pre_labels[i, j, k] == 0 and actual_labels[i, j, k] == 1:
                            FN = FN + 1
        Se.append(float(TP) / (TP + FN))
        Sp.append(float(TN) / (TN + FP))
        Acc.append(float(TP + TN) / (TN + TP + FN + FP))
        dice.append(2 * TP / (2 * TP + FP + FN))

    return np.mean(Se), np.mean(Sp), np.mean(Acc), np.mean(dice)


def only_fov(imgs, test_border_masks, y_true):
    new_imgs = []
    new_true = []
    y_true = y_true[:, 0:height, 0:width]
    imgs = imgs[:, 0:height, 0:width]
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            for k in range(imgs.shape[2]):
                if test_border_masks[i, 0, j, k] == 1:
                    new_imgs.append(imgs[i, j, k])
                    new_true.append(y_true[i, j, k])
    new_imgs = np.asarray(new_imgs)
    new_true = np.asarray(new_true)
    return new_imgs, new_true


def calc_AUC(pixel_predictions, test_groundTruth, test_border_masks, auc_dir):
    # Pixel
    Pixel_possibility_img = np.copy(pixel_predictions)
    y_true1 = np.copy(test_groundTruth)
    Pixel_possibility_img, y_true1 = only_fov(Pixel_possibility_img, test_border_masks, y_true1)

    Pixel_fpr, Pixel_tpr, Pixel_thresholds = sklearn.metrics.roc_curve((y_true1), Pixel_possibility_img)
    Pixel_AUC_ROC = roc_auc_score(y_true1, Pixel_possibility_img)

    # 画AUC曲线
    roc_curve = plt.figure()
    plt.plot(Pixel_fpr, Pixel_tpr, 'g--', label='Unet (AUC = %0.4f)' % Pixel_AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(auc_dir)
    return Pixel_AUC_ROC


path = 'UNET'
# 下载像素级损失预测的数据
print("下载像素级损失预测的数据")
# 下载模型数据
model = model_from_json(open('./'+path+'/architecture.json').read())
model.load_weights('./'+path+'/best_weight.h5')

# 下载测试数据
test_imgs = load_hdf5("./u_net_data/DRIVE_dataset_imgs_test.hdf5")
test_imgs = np.transpose(test_imgs, (0, 2, 3, 1))
test_groundTruth = load_hdf5("./u_net_data/DRIVE_dataset_groundTruth_test.hdf5")
test_groundTruth = np.transpose(test_groundTruth, (0, 2, 3, 1))
test_groundTruth = np.squeeze(test_groundTruth, axis=-1)
test_border_masks = load_hdf5("./u_net_data/test_masks_boder.hdf5")
test_border_masks = test_border_masks / 255.

# 开始预测
start_time = time.time()
predictions = model.predict(test_imgs, batch_size=32, verbose=1)
end_time = time.time()
print('inference time:', end_time-start_time)
predictions = np.array(predictions)
print("predicted images size :")
print(predictions.shape)

# 恢复原状
test_groundTruth = connect_imgs(test_groundTruth, patch_h, patch_w, new_height, new_width)
predictions = pred_to_imgs(predictions, patch_h, patch_w)
predictions = connect_imgs(predictions, patch_h, patch_w, new_height, new_width)

predictions = predictions[:, 0:height, 0:width]
predictions = predictions[:, 0:height, 0:width]

Pixel_picture = np.copy(predictions)
predictions[np.where(predictions >= 0.50)] = 1
predictions[np.where(predictions < 0.50)] = 0

Pixel_AUC = calc_AUC(Pixel_picture, test_groundTruth, test_border_masks, auc_dir='./'+path+'/AUC.png')

test_groundTruth = test_groundTruth[:, 0:height, 0:width]
iou = np.sum(test_groundTruth * predictions) / (
            np.sum(test_groundTruth) + np.sum(predictions) - np.sum(test_groundTruth * predictions))
# # 生成图像
full_imgs = np.uint8(Pixel_picture * 255)
for i in range(Nimgs):
    prediction = Image.fromarray(full_imgs[i], mode="L")
    prediction.save('./'+path+'/vis/' + str(i) + "_PIL_precdiction.png")

# 计算TP,TN,FP,FN
print("正在计算评价指标：")
Pixel_Se, Pixel_Sp, Pixel_Acc, Pixel_F1_Score, dice = calc_Evaluation(predictions, test_groundTruth, test_border_masks)
print("Pixel_Se,Pixel_Sp,Pixel_Acc,Pixel_AUC,Pixel_F1score, dice, iou:", Pixel_Se, Pixel_Sp, Pixel_Acc, Pixel_AUC,
      Pixel_F1_Score, dice, iou)


mean_Se, mean_Sp, mean_Acc, mean_dice = calc_mean_dice(Pixel_picture, test_groundTruth, test_border_masks)

with open('./'+path+'/seg_performance.txt', 'w') as f:
    f.write('SE, SP, ACC, AUC, F1, dice, iou: ')
    f.write(str(Pixel_Se) + ' ')
    f.write(str(Pixel_Sp) + ' ')
    f.write(str(Pixel_Acc) + ' ')
    f.write(str(Pixel_AUC) + ' ')
    f.write(str(Pixel_F1_Score) + '  ')
    f.write(str(dice) + ' ')
    f.write(str(iou) + '    ')
    f.write('mean_SE, mean_SP, mean_Acc, mean_dice: ')
    f.write(str(mean_Se) + ' ')
    f.write(str(mean_Sp) + ' ')
    f.write(str(mean_Acc) + ' ')
    f.write(str(mean_dice) + '    ')
    f.write('inference time: ')
    f.write(str(end_time - start_time))
