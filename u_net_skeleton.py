# coding:utf-8

import numpy as np
import cv2
from numpy import *
# DRIVE
Nimgs=20
height=584
width=565
new_height=640
new_width=640
channels=3
patch_h=64
patch_w=64
per_img_patch_count=3000
Max_Length=10

# STARE
# Nimgs=10
# height=605
# width=700
# new_height=704
# new_width=704
# channels=3
# patch_h=48
# patch_w=48
# per_img_patch_count=6000
# Max_Length=10

# CHASEDB
# Nimgs=20
# height=960
# width=999
# new_height=1024
# new_width=1024
# channels=3
# patch_h=64
# patch_w=64
# per_img_patch_count=4800
# Max_Length=10


# 骨架提取(输入尺寸：[高,宽])
def skeleton(imgs):
    assert(len(imgs.shape)==2)
    P=np.empty(8)
    while(True):
        dex_1 = np.array(np.where(imgs == 1))
        flag=0
        delete_list=[]
        for i in range(dex_1.shape[1]):
            B_P1 = 0
            # imgs[dex_1[0,i]-1,dex_1[1,i]-1]  imgs[dex_1[0,i]-1,dex_1[1,i]]   imgs[dex_1[0,i]-1,dex_1[1,i]+1],   0  1  2
            # imgs[dex_1[0,i],dex_1[1,i]-1]    imgs[dex_1[0,i],dex_1[1,i]]     imgs[dex_1[0,i],dex_1[1,i]+1]      7     3
            # imgs[dex_1[0,i]+1,dex_1[1,i]-1]  imgs[dex_1[0,i]+1,dex_1[1,i]]   imgs[dex_1[0,i]+1,dex_1[1,i]+1]    6  5  4
            if dex_1[0,i]>=1 and dex_1[1,i]>=1 and dex_1[0,i]<=imgs.shape[0]-2 and dex_1[1,i]<=imgs.shape[1]-2:
                P[0],P[1],P[2]=imgs[dex_1[0, i] - 1, dex_1[1, i] - 1], imgs[dex_1[0, i] - 1, dex_1[1, i]], imgs[dex_1[0, i] - 1,dex_1[1, i] + 1]
                P[3], P[4], P[5]=imgs[dex_1[0, i], dex_1[1, i] + 1], imgs[dex_1[0, i] + 1, dex_1[1, i] + 1], imgs[dex_1[0, i] + 1,dex_1[1, i]]
                P[6], P[7] = imgs[dex_1[0, i] + 1, dex_1[1, i] - 1],imgs[dex_1[0, i] , dex_1[1, i] - 1]
                B_P1=sum(P)
                if B_P1>=2 and B_P1<=6:
                    A_P1 = 0
                    for j in range(P.shape[0]):
                        if j<7 and P[j]==0 and P[j+1]==1:
                            A_P1 = A_P1 + 1
                        if j==7 and P[7]==0 and P[0]==1:
                            A_P1 = A_P1 + 1
                    if A_P1==1:
                        if P[1]*P[3]*P[5]==0 and P[3]*P[5]*P[7]==0:
                            flag=1
                            delete_list.append([dex_1[0, i], dex_1[1, i]])
                            # imgs[dex_1[0, i], dex_1[1, i]]=0
        if flag==0:
            break
        for i in range(len(delete_list)):
            imgs[delete_list[i][0],delete_list[i][1]] = 0
        flag=0
        delete_list = []
        dex_1 = np.array(np.where(imgs == 1))
        for i in range(dex_1.shape[1]):
            B_P1=0
            if dex_1[0, i] >= 1 and dex_1[1, i] >= 1 and dex_1[0, i] <= imgs.shape[0] - 2 and dex_1[1, i] <= imgs.shape[1] - 2:
                P[0], P[1], P[2] = imgs[dex_1[0, i] - 1, dex_1[1, i] - 1], imgs[dex_1[0, i] - 1, dex_1[1, i]], imgs[
                    dex_1[0, i] - 1, dex_1[1, i] + 1]
                P[3], P[4], P[5] = imgs[dex_1[0, i], dex_1[1, i] + 1], imgs[dex_1[0, i] + 1, dex_1[1, i] + 1], imgs[
                    dex_1[0, i] + 1, dex_1[1, i]]
                P[6], P[7] = imgs[dex_1[0, i] + 1, dex_1[1, i] - 1], imgs[dex_1[0, i], dex_1[1, i] - 1]
                B_P1 = sum(P)
                if B_P1 >= 2 and B_P1 <= 6:
                    A_P1 = 0
                    for j in range(P.shape[0]):
                        if j < 7 and P[j] == 0 and P[j + 1] == 1:
                            A_P1 = A_P1 + 1
                        if j == 7 and P[7] == 0 and P[0] == 1:
                            A_P1 = A_P1 + 1
                    if A_P1 == 1:
                        if P[1] * P[3] * P[7] == 0 and P[1] * P[5] * P[7] == 0:
                            imgs[dex_1[0, i], dex_1[1, i]] = 0
                            flag=1
        if flag==0:
            break
        for i in range(len(delete_list)):
            imgs[delete_list[i][0],delete_list[i][1]] = 0
    return imgs


# 探寻某点8领域内配置
def find_8_field(imgs,dex,old_point=[]):
    dex=np.array(dex)
    P=np.empty(8)
    point_1=[]
    count_1=0
    # imgs[dex[0]-1,dex[1]-1]  imgs[dex[0]-1,dex[1]]   imgs[dex[0]-1,dex[1]+1],   0  1  2
    # imgs[dex[0],dex[1]-1]    imgs[dex[0],dex[1]]     imgs[dex[0],dex[1]+1]      7     3
    # imgs[dex[0]+1,dex[1]-1]  imgs[dex[0]+1,dex[1]]   imgs[dex[0]+1,dex[1]+1]    6  5  4
    if dex[0] >= 1 and dex[1] >= 1 and dex[0] <= imgs.shape[0] - 2 and dex[1] <= imgs.shape[1] - 2:
        P[0], P[1], P[2] = imgs[dex[0] - 1, dex[1] - 1], imgs[dex[0] - 1, dex[1]], imgs[dex[0] - 1, dex[1]  + 1]
        P[3], P[4], P[5] = imgs[dex[0], dex[1]  + 1], imgs[dex[0] + 1, dex[1]  + 1], imgs[dex[0] + 1, dex[1] ]
        P[6], P[7] = imgs[dex[0] + 1, dex[1]  - 1], imgs[dex[0], dex[1]  - 1]
        count_1=(abs(P[0]-P[1])+abs(P[1]-P[2])+abs(P[2]-P[3])+abs(P[3]-P[4])+abs(P[4]-P[5])+abs(P[5]-P[6])+abs(P[6]-P[7])+abs(P[7]-P[0]))/2
        if P[1] == 1 and [dex[0] - 1, dex[1]] not in old_point: point_1.append([dex[0] - 1, dex[1]])
        if P[3] == 1 and [dex[0], dex[1] + 1] not in old_point: point_1.append([dex[0], dex[1] + 1])
        if P[5] == 1 and [dex[0] + 1, dex[1]] not in old_point: point_1.append([dex[0] + 1, dex[1]])
        if P[7] == 1 and [dex[0], dex[1] - 1] not in old_point: point_1.append([dex[0], dex[1] - 1])
        if P[0]==1 and P[1] == 0 and P[7] == 0 and [dex[0] - 1, dex[1] - 1] not in old_point: point_1.append([dex[0] - 1, dex[1] - 1])
        if P[2]==1 and P[1] == 0 and P[3] == 0 and [dex[0]-1,dex[1]+1] not in old_point: point_1.append([dex[0]-1,dex[1]+1])
        if P[4] == 1 and P[3] == 0 and P[5] == 0 and [dex[0]+1,dex[1]+1] not in old_point: point_1.append([dex[0]+1,dex[1]+1])
        if P[6] == 1 and P[5] == 0 and P[7] == 0 and [dex[0]+1,dex[1]-1] not in old_point: point_1.append([dex[0]+1,dex[1]-1])

    return point_1,count_1


# 通过检测交叉像素将骨架分段Si
def fenduan_Si(imgs):
    assert (len(imgs.shape) == 2)
    P = np.empty(8)
    dex_1 = np.array(np.where(imgs == 1))
    cross_point=[]
    for i in range(dex_1.shape[1]):
        # 判断该点是否为交叉点
        point_1,count_1=find_8_field(imgs,[dex_1[0,i],dex_1[1,i]])
        if count_1==3 or count_1==4:
            cross_point.append([dex_1[0,i],dex_1[1,i]])
        elif dex_1[0, i] == 1:
            cross_point.append([dex_1[0, i], dex_1[1, i]])
        elif dex_1[1, i] == 1:
            cross_point.append([dex_1[0, i], dex_1[1, i]])
        elif dex_1[0, i] == imgs.shape[0] - 2:
            cross_point.append([dex_1[0, i], dex_1[1, i]])
        elif dex_1[1, i] == imgs.shape[1] - 2:
            cross_point.append([dex_1[0, i], dex_1[1, i]])
    # 找到当前图像所有交叉点后，开始从每一个交叉点寻找枝杈
    # if cross_point==[]:   # 如果没有交叉点，选取图像边缘点作为交叉点
    #     for i in range(dex_1.shape[1]):
    #         if dex_1[0,i]==1:
    #             cross_point.append([dex_1[0,i],dex_1[1,i]])
    #         elif dex_1[1,i]==1:
    #             cross_point.append([dex_1[0, i], dex_1[1, i]])
    #         elif dex_1[0, i] == imgs.shape[0]-2:
    #             cross_point.append([dex_1[0, i], dex_1[1, i]])
    #         elif dex_1[1,i]==imgs.shape[1]-2:
    #             cross_point.append([dex_1[0, i], dex_1[1, i]])
    duan_img={}
    itor=0
    duan = []
    old_point = []
    for i in range(len(cross_point)):
        point_1, count_1 = find_8_field(imgs, [cross_point[i][0], cross_point[i][1]],old_point)
        old_point.append([cross_point[i][0], cross_point[i][1]])
        for m in range(len(point_1)): old_point.append([point_1[m][0], point_1[m][1]])
        for j in range(len(point_1)):    # 对交叉点附近为一的点分别进行寻找
            if len(duan)!=0:                                        # 将每一段的坐标存储起来
                if len(duan)>Max_Length:                            # 判断当前段长度是否大于最大段长度
                    for m in range(int(len(duan)/Max_Length)):      # 将当前段切分为几个小段
                        duan_img[itor]=duan[0:Max_Length][:]
                        itor = itor + 1
                        del duan[0:Max_Length]
                    if len(duan)!=0:
                        duan_img[itor] = duan
                        itor = itor + 1
                else:
                    duan_img[itor]=duan
                    itor = itor + 1
            duan=[]                      # 存储每一段元素的下标
            point=[]
            point.append([point_1[j][0], point_1[j][1]])
            break_flag=0
            while(True):
                for k in range(len(point)):
                    point_11,count_11=find_8_field(imgs,[point[k][0],point[k][1]])
                    if count_11==2:          # 判断该点是否属于内点
                        duan.append([point[k][0],point[k][1]])
                        old_point.append([point[k][0],point[k][1]])
                    else:   # 追寻到边界点
                        break_flag=1
                if break_flag==1:
                    break
                point=[]
                for k in range(len(point_11)):    # 提取附近为1的新点
                    if [point_11[k][0],point_11[k][1]] not in old_point:
                        point.append([point_11[k][0],point_11[k][1]])
                if point==[]:
                    break
    return duan_img


def delete_error(duan_Vi):
    length=0
    correct_duan=[]
    for i in duan_Vi:
        length=len(i)+length
    average=length/len(duan_Vi)
    for i in range(len(duan_Vi)):
        if len(duan_Vi[i])-average<5:
            correct_duan.append(duan_Vi[i])
    return correct_duan


def Find_Vi(duan_img,original_img):
    Vi_dex={}               # 存储原始图像的所有像素段
    itor=0
    duan_Vi = []            # 存储原始图像的像素段
    duan = []
    old_x,old_y=0,0         # 存储上一次循环的坐标
    for i in range(len(duan_img)):
        if duan_Vi!=[]:
            Vi_dex[itor]=delete_error(duan_Vi)
            itor=itor+1
        duan_Vi = []
        # 计算当前段的斜率，如果斜率大于45度则横向寻找，反之则纵向寻找
        if abs(duan_img[i][-1][1] - duan_img[i][0][1])!=0:
            K = abs(duan_img[i][-1][0] - duan_img[i][0][0]) / float(abs(duan_img[i][-1][1] - duan_img[i][0][1]))
        else:
            K=2
        for j in duan_img[i]:
            duan=[]
            duan.append([j[0], j[1]])
            if K>1:   # 横向寻找
                x1,x2=j[1]-1,j[1]+1
                y1,y2=j[0],j[0]
                Left_break_flag,Right_break_flag=0,0
                while(True):
                    if j[0]==old_y:
                        duan=[]
                        break
                    if original_img[y1][x1]==1 and Left_break_flag==0:
                        duan.append([y1,x1])
                    else:
                        Left_break_flag=1
                    if original_img[y2][x2]==1 and Right_break_flag==0:
                        duan.append([y2,x2])
                    else:
                        Right_break_flag=1
                    x1, x2 = x1 - 1, x2 + 1
                    if Left_break_flag==1 and Right_break_flag==1 or x1<0 or x2>original_img.shape[1]-1:
                        old_y=j[0]
                        break
            else:      # 纵向寻找
                x1, x2 = j[1],j[1]
                y1, y2 = j[0]-1,j[0]+1
                Up_break_flag, Down_break_flag = 0, 0
                while (True):
                    if j[1]==old_x:
                        duan=[]
                        break
                    if original_img[y1][x1] == 1 and Up_break_flag == 0:
                        duan.append([y1,x1])
                    else:
                        Up_break_flag = 1
                    if original_img[y2][x2] == 1 and Down_break_flag == 0:
                        duan.append([y2,x2])
                    else:
                        Down_break_flag = 1
                    y1, y2 = y1 - 1, y2 + 1
                    if Up_break_flag == 1 and Down_break_flag == 1 or y1<0 or y2>original_img.shape[0]-1:
                        old_x=j[1]
                        break
            if duan!=[]:
                duan_Vi.append(duan)
    # 计算Vi
    Vi=[]
    for i in range(len(Vi_dex)):
        length=0
        for j in range(len(Vi_dex[i])):
            length=length+len(Vi_dex[i][j])
        Vi.append(length)
    # 计算T_Vi
    T_Vi=[]
    for i in range(len(Vi)):
        T_Vi.append(float(Vi[i])/len(duan_img[i]))
    return Vi_dex,T_Vi


# 生成权重图
def generate_weight_map(train_groundTruth,prediction_train_groundTruth,R=None):
    assert (len(train_groundTruth.shape)==3 and len(prediction_train_groundTruth.shape)==3)
    assert (train_groundTruth.shape[0]==prediction_train_groundTruth.shape[0])
    weight=np.zeros((train_groundTruth.shape[0],train_groundTruth.shape[1],train_groundTruth.shape[2]))
    New_train_groundTruth=train_groundTruth.copy()
    X_list,Y_list=[],[]
    for i in range(New_train_groundTruth.shape[0]):
        New_train_groundTruth[i]=skeleton(New_train_groundTruth[i])
        train_duan_img=fenduan_Si(New_train_groundTruth[i])
        train_Vi_dex,train_T_Vi=Find_Vi(train_duan_img,train_groundTruth[i])
        for j in range(len(train_Vi_dex)):
            X_list=[n[0] for m in train_Vi_dex[j] for n in m]
            Y_list=[n[1] for m in train_Vi_dex[j] for n in m]
            h=max(X_list)-min(X_list)
            w=max(Y_list)-min(Y_list)
            y=int((max(X_list)+min(X_list))/2)
            x=int((max(Y_list)+min(Y_list))/2)
            prediction_Vi_dex = np.array(np.where(prediction_train_groundTruth[i, y - int(np.ceil(h / 2)):y +
                              int(np.ceil(h / 2)+1), x - int(np.ceil(w / 2)): x + int(np.ceil(w / 2)+1)] == 1))
            prediction_T_Vi=prediction_Vi_dex.shape[1]/float(len(train_duan_img[j]))
            MR=abs(prediction_T_Vi-train_T_Vi[j])/train_T_Vi[j]

            if prediction_T_Vi==0:
                for k in range(len(train_Vi_dex[j])):
                    for l in train_Vi_dex[j][k]:
                        weight[i,l[0],l[1]]=1+MR
            else:
                for k in range(prediction_Vi_dex.shape[1]):
                    weight[i,prediction_Vi_dex[0,k]+y - int(np.ceil(h / 2)),prediction_Vi_dex[1,k]+x - int(np.ceil(w / 2))]=1+MR

    zeros_dex=np.array(np.where(weight==0))
    for i in range(zeros_dex.shape[1]):
        weight[zeros_dex[0,i],zeros_dex[1,i],zeros_dex[2,i]]=1
    return weight


# 把one——hot解码
def pred_to_imgs(pred, patch_height=patch_h, patch_width=patch_w):
    assert (len(pred.shape)==2)
    # assert (pred.shape[2]==2)
    final_img=[]
    # pred_images = pred[:, :, 1]
    pred_images = pred
    # pred_images = np.reshape(pred_images, (pred_images.shape[0], 48, 48))

    pred_images = uint8(pred_images * 255.0)
    N = pred_images.shape[0]
    for n in range(N):
        ret1, th1 = cv2.threshold(pred_images[n], 0, 255, cv2.THRESH_OTSU)
        th1[np.where(th1==255)]=1
        final_img.append(th1)
    final_img=np.array(final_img)
    return final_img
    # thresold=calc_best_thresold(pred)
    # pred_images = np.empty((pred.shape[0],pred.shape[1]))
    # for i in range(pred.shape[0]):
    #     for pix in range(pred.shape[1]):
    #         if pred[i,pix,1]>=pred[i,pix,0]:
    #             pred_images[i,pix]=1
    #         else:
    #             pred_images[i,pix]=0
    # pred_images = np.reshape(pred_images,(pred_images.shape[0], patch_height, patch_width))
    # return pred_images


def calc_best_thresold(imgs):
    # assert (len(imgs.shape) == 3 and imgs.shape[2] == 2)

    # pred_images = np.empty((imgs.shape[0], imgs.shape[1]))
    # pred_images = imgs[:, :, 1]
    # for i in range(imgs.shape[0]):
    #     for pix in range(imgs.shape[1]):
    #         pred_images[i, pix] = imgs[i, pix, 1]
    # pred_images = np.reshape(pred_images, (pred_images.shape[0],64, 64))

    imgs = imgs * 255.0
    # N = imgs.shape[0]
    # best_thresold=[]
    # for n in range(N):
    ret1, th1=cv2.threshold(imgs, 0, 255, cv2.THRESH_OTSU)
    best_thresold= ret1/255.0
        # best_thresold.append(OTSU(pred_images[n]))
    # imgs = imgs * 255.0
    # N = imgs.shape[0]
    # best_thresold=[]
    # for n in range(N):
    #    best_thresold.append(OTSU(imgs[n]))
    return best_thresold


def OTSU(img_array):                 # 传入的参数为ndarray形式
    img_array=img_array*255.0
    height = img_array.shape[0]
    width = img_array.shape[1]
    if height * width > 0:
        count_pixel = np.zeros(256)

        for i in range(height):
            for j in range(width):
                count_pixel[int(img_array[i][j])] += 1

        max_variance = 0.0
        best_thresold = 0
        for thresold in range(256):
            n0 = count_pixel[:thresold].sum()
            n1 = count_pixel[thresold:].sum()
            w0 = n0 / (height * width)
            w1 = n1 / (height * width)
            u0 = 0.0
            u1 = 0.0

            for i in range(thresold):
                u0 += i * count_pixel[i]
            for j in range(thresold, 256):
                u1 += j * count_pixel[j]

            u = u0 * w0 + u1 * w1
            tmp_var = w0 * np.power((u - u0), 2) + w1 * np.power((u - u1), 2)

            if tmp_var > max_variance:
                best_thresold = thresold
                max_variance = tmp_var

        best_thresold=best_thresold/255.0
        return best_thresold
    else:
        return None


def generate_width_map(train_groundTruth):
    weight=np.ones_like(train_groundTruth)*100
    # 测试
    for l in range(train_groundTruth.shape[0]):
        label = train_groundTruth[l]
        dex_1 = np.array(np.where(label == 1))
        for i in range(dex_1.shape[1]):
            x_count = 1
            y_count = 1
            x_edge = []
            y_edge = []
            x_point=[]
            y_point=[]
            y = dex_1[0, i]
            x = dex_1[1, i]
            break_flag1, break_flag2 = 0, 0
            x1, x2 = x, x
            y1, y2 = y, y
            x_point.append([y,x])
            y_point.append([y,x])
            while (True):
                if break_flag1 != 1:
                    x1 = x1 + 1
                if dex_1[0, i]<label.shape[0] and x1<label.shape[1] and label[dex_1[0, i], x1] == 1:
                    x_point.append([dex_1[0, i], x1])
                    x_count = x_count + 1
                elif break_flag1!=1 :
                    x_edge.append([dex_1[0, i], x1 - 1])
                    break_flag1 = 1
                if break_flag2 != 1:
                    x2 = x2 - 1
                if label[dex_1[0, i], x2] == 1:
                    x_point.append([dex_1[0, i], x2])
                    x_count = x_count + 1
                elif break_flag2 != 1:
                    x_edge.append([dex_1[0, i], x2 + 1])
                    break_flag2 = 1
                if break_flag1 == 1 and break_flag2 == 1:
                    break_flag1, break_flag2 = 0, 0
                    break
            x1, x2 = x, x
            y1, y2 = y, y
            while (True):
                if break_flag1 != 1:
                    y1 = y1 + 1
                if y1 < label.shape[0] and dex_1[1, i]<label.shape[1] and label[y1, dex_1[1, i]] == 1:
                    y_point.append([y1, dex_1[1, i]])
                    y_count = y_count + 1
                elif break_flag1 != 1:
                    y_edge.append([y1 - 1, dex_1[1, i]])
                    break_flag1 = 1
                if break_flag2 != 1:
                    y2 = y2 - 1
                if label[y2, dex_1[1, i]] == 1:
                    y_point.append([y2, dex_1[1, i]])
                    y_count = y_count + 1
                elif break_flag2 != 1:
                    y_edge.append([y2 + 1, dex_1[1, i]])
                    break_flag2 = 1
                if break_flag1 == 1 and break_flag2 == 1:
                    break_flag1, break_flag2 = 0, 0
                    break
            if x_count>=y_count and y_count<weight[l,y_point[0][0],y_point[0][1]]:
                for j in y_point:
                    if y_count<weight[l,j[0],j[1]]:
                        weight[l,j[0],j[1]]=y_count
                # x1, y1 = y_edge[0][1], y_edge[0][0]
                # x2, y2 = y_edge[1][1], y_edge[1][0]
                # y_count = 1000+y_count    ############################################################
                # weight[l, y2, x2] = y_count
                # weight[l, y1, x1] = y_count
                # if y2 >= y1:
                #     if np.sum(weight[l, y2 + 1:y2 + 4, x2]) == 300:
                #         weight[l, y2 + 1, x2] = y_count
                #         weight[l, y2 + 2, x2] = y_count
                #         # weight[l, y2 + 3, x2] = y_count
                #         # weight[l, y2 + 4, x2] = y_count
                #     if np.sum(weight[l, y1 - 3:y1, x2]) == 300:
                #         weight[l, y1 - 1, x1] = y_count
                #         weight[l, y1 - 2, x1] = y_count
                #         # weight[l, y1 - 3, x1] = y_count
                #         # weight[l, y1 - 4, x1] = y_count
                # else:
                #     if np.sum(weight[l, y2 - 3:y2 , x2]) == 300:
                #         weight[l, y2 - 1, x2] = y_count
                #         weight[l, y2 - 2, x2] = y_count
                #         # weight[l, y2 - 3, x2] = y_count
                #         # weight[l, y2 - 4, x2] = y_count
                #     if np.sum(weight[l, y1 + 1:y1 + 4, x2]) == 300:
                #         weight[l, y1 + 1, x1] = y_count
                #         weight[l, y1 + 2, x1] = y_count
                #         # weight[l, y1 + 3, x1] = y_count
                #         # weight[l, y1 + 4, x1] = y_count
            elif x_count<weight[l,x_point[0][0],x_point[0][1]]:
                for j in x_point:
                    if x_count<weight[l,j[0],j[1]]:
                        weight[l,j[0],j[1]]=x_count
                # x1, y1 = x_edge[0][1], x_edge[0][0]
                # x2, y2 = x_edge[1][1], x_edge[1][0]
                # x_count=1000+x_count            ###########################################
                # weight[l, y2, x2] = x_count
                # weight[l, y1, x1] = x_count
                # if x2 >= x1:
                #     if np.sum(weight[l, y2, x2 +1: x2 +4]) == 300 :
                #         weight[l, y2, x2+1] = x_count
                #         weight[l, y2, x2+2] = x_count
                #         # weight[l, y2, x2+3] = x_count
                #         # weight[l, y2, x2+4] = x_count
                #     if np.sum(weight[l, y2, x1 - 3: x1 ]) == 300:
                #         weight[l, y1, x1-1] = x_count
                #         weight[l, y1, x1-2] = x_count
                #         # weight[l, y1, x1-3] = x_count
                #         # weight[l, y1, x1-4] = x_count
                # else:
                #     if np.sum(weight[l, y2, x2 - 3: x2 ]) == 300 :
                #         weight[l, y2, x2-1] = x_count
                #         weight[l, y2, x2-2] = x_count
                #         # weight[l, y2, x2-3] = x_count
                #         # weight[l, y2, x2-4] = x_count
                #     if np.sum(weight[l, y2, x1 + 1: x1 + 4]) == 300:
                #         weight[l, y1, x1+1] = x_count
                #         weight[l, y1, x1+2] = x_count
                #         # weight[l, y1, x1+3] = x_count
                #         # weight[l, y1, x1+4] = x_count
    weight[np.where(weight==100)] = 0.0
    return weight
    #         if x_count > y_count and y_count < 8:  # y方向扩充
    #             increase_count = 8 - y_count
    #             x1, y1 = y_edge[0][1], y_edge[0][0]
    #             x2, y2 = y_edge[1][1], y_edge[1][0]
    #             if y2 >= y1:
    #                 if x1 == x2 and y1 == y2:
    #                     data = imgs[y1, x1]
    #                 else:
    #                     data = np.mean(imgs[y1:y2, x1])
    #                 retain_feature = []
    #                 retain_feature.append(imgs[y1 - 1, x1])
    #                 retain_feature.append(imgs[y1 - 2, x1])
    #                 retain_feature.append(imgs[y2 + 1, x2])
    #                 retain_feature.append(imgs[y2 + 2, x2])
    #                 for k in range(int(increase_count / 2.0)):
    #                     y2 = y2 + 1
    #                     imgs[y2, x2] = data
    #                     label[y2, x2] = 1.0
    #                     y1 = y1 - 1
    #                     imgs[y1, x1] = data
    #                     label[y1, x1] = 1.0
    #                 for k in range(2):
    #                     y2 = y2 + 1
    #                     y1 = y1 - 1
    #                     imgs[y1, x1] = retain_feature[k]
    #                     imgs[y2, x2] = retain_feature[k + 2]
    #             else:
    #                 data = np.mean(imgs[y2:y1, x1])
    #                 retain_feature = []
    #                 retain_feature.append(imgs[y1 + 1, x1])
    #                 retain_feature.append(imgs[y1 + 2, x1])
    #                 retain_feature.append(imgs[y2 - 1, x2])
    #                 retain_feature.append(imgs[y2 - 2, x2])
    #                 for k in range(int(increase_count / 2.0)):
    #                     y2 = y2 - 1
    #                     imgs[y2, x2] = data
    #                     label[y2, x2] = 1.0
    #                     y1 = y1 + 1
    #                     imgs[y1, x1] = data
    #                     label[y1, x1] = 1.0
    #                 for k in range(2):
    #                     y2 = y2 - 1
    #                     y1 = y1 + 1
    #                     imgs[y1, x1] = retain_feature[k]
    #                     imgs[y2, x2] = retain_feature[k + 2]
    #         else:  # x方向扩充
    #             increase_count = 8 - x_count
    #             x1, y1 = x_edge[0][1], x_edge[0][0]
    #             x2, y2 = x_edge[1][1], x_edge[1][0]
    #             if x2 >= x1:
    #                 if x1 == x2 and y1 == y2:
    #                     data = imgs[y1, x1]
    #                 else:
    #                     data = np.mean(imgs[y1, x1:x2])
    #                 retain_feature = []
    #                 retain_feature.append(imgs[y1, x1 - 1])
    #                 retain_feature.append(imgs[y1, x1 - 2])
    #                 retain_feature.append(imgs[y2, x2 + 1])
    #                 retain_feature.append(imgs[y2, x2 + 2])
    #                 for k in range(int(increase_count / 2.0)):
    #                     x2 = x2 + 1
    #                     imgs[y2, x2] = data
    #                     label[y2, x2] = 1.0
    #                     x1 = x1 - 1
    #                     imgs[y1, x1] = data
    #                     label[y1, x1] = 1.0
    #                 for k in range(2):
    #                     x2 = x2 + 1
    #                     x1 = x1 - 1
    #                     imgs[y1, x1] = retain_feature[k]
    #                     imgs[y2, x2] = retain_feature[k + 2]
    #             else:
    #                 data = np.mean(imgs[y1, x2:x1])
    #                 retain_feature = []
    #                 retain_feature.append(imgs[y1, x1 + 1])
    #                 retain_feature.append(imgs[y1, x1 + 2])
    #                 retain_feature.append(imgs[y2, x2 - 1])
    #                 retain_feature.append(imgs[y2, x2 - 2])
    #                 for k in range(int(increase_count / 2.0)):
    #                     x2 = x2 - 1
    #                     imgs[y2, x2] = data
    #                     label[y2, x2] = 1.0
    #                     x1 = x1 + 1
    #                     imgs[y1, x1] = data
    #                     label[y1, x1] = 1.0
    #                 for k in range(2):
    #                     x2 = x2 - 1
    #                     x1 = x1 + 1
    #                     imgs[y1, x1] = retain_feature[k]
    #                     imgs[y2, x2] = retain_feature[k + 2]
    #     # plt.imshow(train_imgs[0])
    #     # plt.axis('off')
    #     # plt.show()
    #     # plt.imshow(train_groundTruth[0])
    #     # plt.axis('off')
    #     # plt.show()
    #     # exit()
    #
    # return train_imgs, train_groundTruth


# def increase_weight_width(out_line,weight):
#     out_line=out_line/255.0
#     new_weight = np.copy(weight)
#     # 测试
#     for l in range(out_line.shape[0]):
#         single_out_line=out_line[l]
#         dex_1 = np.array(np.where(single_out_line == 1))
#         for i in range(dex_1.shape[1]):
#             y = dex_1[0, i]
#             x = dex_1[1, i]
