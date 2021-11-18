# coding:utf-8

import matplotlib.pyplot as plt # plt 用于显示图片
import numpy as np
import h5py
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sklearn.metrics
from skimage.measure import label
import skeleton_width

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


def write_hdf5(arr,fpath):
  with h5py.File(fpath,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


def load_hdf5(infile):
  with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
    return f["image"][()]


def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    # assert (pred.shape[2]==2 )  #check the classes are 2
    pred_images = np.empty((pred.shape[0],pred.shape[1]))  #(Npatches,height*width)
    if mode=="original":
        if mode == "original":
            for i in range(pred.shape[0]):
                for pix in range(pred.shape[1]):
                    pred_images[i, pix] = pred[i, pix, 1]
    elif mode=="Pixel":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>=0.5:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    elif mode=="Joint":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,0]>=0.5:
                    pred_images[i,pix]= pred[i, pix, 1]
                elif pred[i,pix,1]>=0.5:
                    pred_images[i,pix]=pred[i, pix, 1]
                elif pred[i,pix,2]>=0.5:
                    pred_images[i,pix]= pred[i, pix, 1]
                # elif pred[i,pix,3]==1:
                #     pred_images[i,pix]=3
                # elif pred[i,pix,4]==1:
                #     pred_images[i,pix]=4
                # elif pred[i,pix,5]==1:
                #     pred_images[i,pix]=5
                # else :
                #     pred_images[i,pix]=6

    else:
        print("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images,(pred_images.shape[0], patch_height, patch_width))
    return pred_images


def connect_imgs(imgs,patch_h,patch_w,height,width):
    itor=0
    new_imgs=np.empty((Nimgs,height,width))
    for k in range(Nimgs):
        for i in range(int(height/patch_h)):
            hang_imgs = imgs[itor]
            for j in range(int(width/patch_w)-1):
                hang_imgs=np.concatenate((hang_imgs,imgs[itor+1]),axis=1)
                itor=itor+1
            if i==0:
                itor=itor+1
                lie_imgs =hang_imgs
            else:
                itor=itor+1
                lie_imgs=np.concatenate((lie_imgs,hang_imgs),axis=0)
        new_imgs[k]=lie_imgs
    assert(new_imgs.shape==(Nimgs,height,width))
    return new_imgs


# 计算TP,TN,FP,FN
def calc_Evaluation(pre_labels,actual_labels,test_border_masks):
    pre_labels=pre_labels[:,0:height,0:width]
    actual_labels = actual_labels[:, 0:height, 0:width]
    TP,TN,FP,FN=0,0,0,0
    per_TP,per_TN,per_FP,per_FN=0,0,0,0
    per_Se,per_Sp,per_Acc=[],[],[]
    for i in range(pre_labels.shape[0]):
        per_TP, per_TN, per_FP, per_FN = 0, 0, 0, 0
        for j in range(pre_labels.shape[1]):
            for k in range(pre_labels.shape[2]):
                if test_border_masks[i,0,j,k]==1:
                    if pre_labels[i,j,k]==1 and actual_labels[i,j,k]==1:
                        TP=TP+1
                        per_TP=per_TP+1
                    elif pre_labels[i, j, k] == 0 and actual_labels[i, j, k] == 0:
                        TN = TN + 1
                        per_TN = per_TN + 1
                    elif pre_labels[i, j, k] == 1 and actual_labels[i, j, k] == 0:
                        FP = FP + 1
                        per_FP = per_FP + 1
                    elif pre_labels[i, j, k] == 0 and actual_labels[i, j, k] == 1:
                        FN = FN + 1
                        per_FN = per_FN + 1
        per_Se.append(float(per_TP)/(per_TP+per_FN))
        per_Sp .append(float(per_TN) / (per_TN + per_FP))
        per_Acc.append(float(per_TP + per_TN) / (per_TN + per_TP + per_FN + per_FP))
    Se=float(TP)/(TP+FN)
    Sp=float(TN)/(TN+FP)
    Acc=float(TP+TN)/(TN + TP + FN + FP)
    Pr=float(TP)/(TP+FP)
    Pixel_F1_Score=2.0*Pr*Se/(Pr+Se)

    mean_Se=np.mean(per_Se)
    mean_Sp = np.mean(per_Sp)
    mean_Acc = np.mean(per_Acc)
    return Se,Sp,Acc,Pr,Pixel_F1_Score,per_Se,per_Sp,per_Acc,mean_Se,mean_Sp,mean_Acc


def only_fov(imgs,test_border_masks,y_true):
    new_imgs=[]
    new_true=[]
    y_true = y_true[:, 0:height, 0:width]
    imgs = imgs[:, 0:height, 0:width]
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            for k in range(imgs.shape[2]):
                if test_border_masks[i,0,j,k]==1:
                    new_imgs.append(imgs[i,j,k])
                    new_true.append(y_true[i, j, k])
    new_imgs=np.asarray(new_imgs)
    new_true = np.asarray(new_true)
    return new_imgs,new_true


def calc_AUC(pixel_predictions,Joint_predictions,test_groundTruth,test_border_masks,auc_dir):
    # Pixel
    Pixel_possibility_img=np.copy(pixel_predictions)

    y_true1 = np.copy(test_groundTruth)
    y_true2 = np.copy(test_groundTruth)
    Pixel_possibility_img,y_true1 = only_fov(Pixel_possibility_img, test_border_masks,y_true1)

    Pixel_fpr, Pixel_tpr, Pixel_thresholds = sklearn.metrics.roc_curve((y_true1), Pixel_possibility_img)
    Pixel_AUC_ROC=roc_auc_score(y_true1, Pixel_possibility_img)
    # Joint
    Joint_possibility_img = np.copy(Joint_predictions)
    Joint_possibility_img, y_true2 = only_fov(Joint_possibility_img, test_border_masks, y_true2)
    Joint_fpr, Joint_tpr, Joint_thresholds = sklearn.metrics.roc_curve((y_true2), Joint_possibility_img)
    Joint_AUC_ROC = roc_auc_score(y_true2, Joint_possibility_img)

    # 画AUC曲线
    roc_curve = plt.figure()
    plt.plot(Pixel_fpr, Pixel_tpr, 'g--', label='Unet (AUC = %0.4f)' % Pixel_AUC_ROC)
    plt.plot(Joint_fpr, Joint_tpr, 'r--', label='Res2_Unet (AUC = %0.4f)' % Joint_AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(auc_dir)
    return Pixel_AUC_ROC,Joint_AUC_ROC


test_groundTruth=load_hdf5("./CHASEDB/u_net_data/CHASEDB_dataset_groundTruth_test.hdf5")
test_groundTruth=np.reshape(test_groundTruth,(int(new_height/patch_h)*int(new_width/patch_w)*Nimgs,patch_h,patch_w))
test_groundTruth=connect_imgs(test_groundTruth,patch_h,patch_w,new_height,new_width)
test_border_masks=load_hdf5("./CHASEDB/u_net_data/test_border_masks.hdf5")

Pixel_predictions=load_hdf5("./CHASEDB/CHASEDB-CHASEDB/UNET和RES2NET(64-64)/UNET/precdiction1.h5")
Pixel=np.copy(Pixel_predictions)
Pixel_predictions = pred_to_imgs(Pixel_predictions, patch_h, patch_w)
Pixel_predictions = connect_imgs(Pixel_predictions, patch_h, patch_w, new_height, new_width)

# 恢复原状
Pixel_predictions = Pixel_predictions[:, 0:height, 0:width]
test_groundTruth = test_groundTruth[:, 0:height, 0:width]

Pixel_picture=np.copy(Pixel_predictions)
Pixel_predictions[np.where(Pixel_predictions >= 0.50)] = 1
Pixel_predictions[np.where(Pixel_predictions < 0.50)] = 0


Pixel_AUC,Joint_AUC = calc_AUC(Pixel_picture, Pixel_picture,test_groundTruth, test_border_masks,
                     auc_dir="./CHASEDB/CHASEDB-CHASEDB/unet和res2net(64-64)/UNET/试验_ROC.png")

# # 生成图像
label=np.copy(test_groundTruth)
Pixel=np.copy(Pixel_predictions)

Picture_1=np.concatenate((label, test_groundTruth), axis=2)
Picture_2=np.concatenate((Pixel, Pixel_picture), axis=2)

full_imgs=np.concatenate((Picture_1, Picture_2), axis=1)


full_imgs=np.uint8(full_imgs*255)
for i in range(Nimgs):
    prediction = Image.fromarray(full_imgs[i], mode="L")
    prediction.save("./CHASEDB/CHASEDB-CHASEDB/unet和res2net(64-64)/UNET/"+str(i)+"_PIL_precdiction.png")


# 计算TP,TN,FP,FN
print("正在计算评价指标：")
Pixel_Se,Pixel_Sp,Pixel_Acc,Pixel_Pr,Pixel_F1_Score,Pixel_per_Se,Pixel_per_Sp,Pixel_per_Acc,\
Pixel_mean_Se,Pixel_mean_Sp,Pixel_mean_Acc=calc_Evaluation(Pixel_predictions,test_groundTruth,test_border_masks)
print("Pixel_Se,Pixel_Sp,Pixel_Acc,Pixel_AUC,Pixel_F1score:",Pixel_Se,Pixel_Sp,Pixel_Acc,Pixel_AUC,Pixel_F1_Score)

