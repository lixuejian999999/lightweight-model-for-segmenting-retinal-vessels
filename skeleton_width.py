#  vessel skeleton
#  rawImg 是标签
import numpy as np

import matplotlib.pyplot as plt  # plt 用于显示图片


# import h5py
# def load_hdf5(infile):
#   with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
#     return f["image"][()]


def getOutlineByMatrix(rawImg):  # 利用周围有无0判断是否是边界
    # rowNum = len(rawImg)     # gao and hang
    # colNum = len(rawImg[0])  # kuan and lie
    rowNum, colNum = rawImg.shape

    newImg = np.ones((rowNum, colNum), np.uint8)

    # print(rowNum,'--',colNum)
    for i in range(rowNum):
        for j in range(colNum):
            if rawImg[i][j] > 0:
                top = i - 1
                bot = i + 1
                lef = j - 1
                rig = j + 1
                if i == 0:
                    top = 0
                if i == rowNum - 1:
                    bot = rowNum - 1
                if j == 0:
                    lef = 0
                if j == colNum - 1:
                    rig = colNum - 1
                # print(i,j,top,bot,lef,rig)
                if rawImg[top][j] == 0 or rawImg[bot][j] == 0 or rawImg[i][lef] == 0 or rawImg[i][rig] == 0:
                    newImg[i][j] = 255
                else:
                    newImg[i][j] = 0
                    # print(rawImg[top][j],rawImg[bot][j],rawImg[i][lef],rawImg[i][rig])
            else:
                newImg[i][j] = 0  # duoyu

    return newImg


def getValueByIndex(i, j, mtx):
    rowNum = len(mtx)
    colNum = len(mtx[0])
    res = 0
    if i >= 0 and i < rowNum and j >= 0 and j < colNum:
        if mtx[i][j] > 0:
            res = 1
    return res


def deleteByOutline(rawImg, Outline):
    rowNum = len(rawImg)
    colNum = len(rawImg[0])

    newImg = np.ones((rowNum, colNum), np.uint8)
    deleteNum = 0

    for i in range(rowNum):
        for j in range(colNum):
            # newImg[i][j] = rawImg[i][j]
            newImg[i][j] = rawImg[i][j]
            if Outline[i][j] > 0:
                outlineNum = 0
                outlineNum2 = 0  # direct
                changeNum = 0

                flag0 = getValueByIndex(i - 1, j - 1, Outline)
                outlineNum = outlineNum + flag0

                flag1 = getValueByIndex(i - 1, j, Outline)
                outlineNum2 = outlineNum2 + flag1
                if flag0 != flag1:
                    changeNum = changeNum + 1

                flag2 = getValueByIndex(i - 1, j + 1, Outline)
                outlineNum = outlineNum + flag2
                if flag1 != flag2:
                    changeNum = changeNum + 1
                flag1 = flag2

                flag2 = getValueByIndex(i, j + 1, Outline)
                outlineNum2 = outlineNum2 + flag2
                if flag1 != flag2:
                    changeNum = changeNum + 1
                flag1 = flag2

                flag2 = getValueByIndex(i + 1, j + 1, Outline)
                outlineNum = outlineNum + flag2
                if flag1 != flag2:
                    changeNum = changeNum + 1
                flag1 = flag2

                flag2 = getValueByIndex(i + 1, j, Outline)
                outlineNum2 = outlineNum2 + flag2
                if flag1 != flag2:
                    changeNum = changeNum + 1
                flag1 = flag2

                flag2 = getValueByIndex(i + 1, j - 1, Outline)
                outlineNum = outlineNum + flag2
                if flag1 != flag2:
                    changeNum = changeNum + 1
                flag1 = flag2

                flag2 = getValueByIndex(i, j - 1, Outline)
                outlineNum2 = outlineNum2 + flag2
                if flag1 != flag2:
                    changeNum = changeNum + 1

                if flag0 != flag2:
                    changeNum = changeNum + 1

                outlineNum = outlineNum2 + outlineNum

                noZeroNum = 0
                noZeroNum2 = 0
                noZeroNum = noZeroNum + getValueByIndex(i - 1, j, rawImg)
                noZeroNum = noZeroNum + getValueByIndex(i + 1, j, rawImg)
                noZeroNum = noZeroNum + getValueByIndex(i, j + 1, rawImg)
                noZeroNum = noZeroNum + getValueByIndex(i, j - 1, rawImg)

                noZeroNum2 = noZeroNum2 + getValueByIndex(i - 1, j - 1, rawImg)
                noZeroNum2 = noZeroNum2 + getValueByIndex(i + 1, j + 1, rawImg)
                noZeroNum2 = noZeroNum2 + getValueByIndex(i - 1, j + 1, rawImg)
                noZeroNum2 = noZeroNum2 + getValueByIndex(i + 1, j - 1, rawImg)
                noZeroNum2 = noZeroNum2 + noZeroNum

                if outlineNum == 2:
                    if outlineNum2 == 0 and noZeroNum > 0:
                        newImg[i][j] = 0  # delete
                        deleteNum = deleteNum + 1
                    if outlineNum2 == 1 and noZeroNum > 1:
                        newImg[i][j] = 0  # delete
                        deleteNum = deleteNum + 1
                    if outlineNum2 == 2 and noZeroNum2 > 2:
                        newImg[i][j] = 0  # delete
                        deleteNum = deleteNum + 1
                if outlineNum == 3:
                    if outlineNum2 == 1 or outlineNum2 == 2:
                        if changeNum == 4:
                            if noZeroNum2 > 3:
                                newImg[i][j] = 0  # delete
                                deleteNum = deleteNum + 1

    return newImg, deleteNum


# outline = getOutlineByMatrix(img)
# skeleton, dn = deleteByOutline(img, outline)
#
# skeleton2 = skeleton
# cnt = 0
# while dn > 0:
#     ol = getOutlineByMatrix(skeleton2)
#     skeleton2, dn = deleteByOutline(skeleton2, ol)
#     # if cnt == 2:
#     #     cv2.imshow('skeleton2', skeleton2)
#     cnt = cnt + 1
#     print(cnt, dn)


#  width of vessel
#  rawImg 是标签  targetImg是骨架


def getWidthByMatrix(rawImg, targetImg):
    rowNum = len(rawImg)  # gao and hang
    colNum = len(rawImg[0])  # kuan and lie

    windowSize = 10

    # newImg = np.ones((rowNum, colNum), np.uint8)
    newImg = np.zeros((rowNum, colNum))
    newImg2 = np.zeros((rowNum, colNum))

    ol = getOutlineByMatrix(rawImg)

    for ti in range(rowNum):
        for tj in range(colNum):
            if targetImg[ti][tj] > 0:

                # dis = rowNum^2 + colNum^2
                dis = windowSize ** 2 + windowSize ** 2
                for ri in range(2 * windowSize + 1):
                    for rj in range(2 * windowSize + 1):
                        # for ri in range(rowNum):
                        #     for rj in range(colNum):
                        wi = ti - windowSize + ri
                        wj = tj - windowSize + rj
                        if wi < 0:
                            wi = 0
                        if wi >= rowNum:
                            wi = rowNum - 1
                        if wj < 0:
                            wj = 0
                        if wj >= colNum:
                            wj = colNum - 1
                        if ol[wi][wj] > 0:
                            tempDis = (wi - ti) ** 2 + (wj - tj) ** 2
                            # if rawImg[ri][rj] > 0:
                            #     tempDis = (ri - ti)^2 + (rj - tj)^2
                            if tempDis < dis:
                                dis = tempDis
                newImg[ti][tj] = dis ** 0.5

                if newImg[ti][tj] < 0.5:
                    newImg[ti][tj] = 0.5

    for ti in range(rowNum):
        for tj in range(colNum):
            if rawImg[ti][tj] > 0:
                # dis = rowNum^2 + colNum^2
                dis = windowSize ** 2 + windowSize ** 2
                mi = 0
                mj = 0
                for ri in range(2 * windowSize + 1):
                    for rj in range(2 * windowSize + 1):
                        wi = ti - windowSize + ri
                        wj = tj - windowSize + rj
                        if wi < 0:
                            wi = 0
                        if wi >= rowNum:
                            wi = rowNum - 1
                        if wj < 0:
                            wj = 0
                        if wj >= colNum:
                            wj = colNum - 1
                        # if ol[wi][wj] > 0:
                        #     tempDis = (ri - ti)^2 + (rj - tj)^2
                        if targetImg[wi][wj] > 0:
                            tempDis = (wi - ti) ** 2 + (wj - tj) ** 2
                            if tempDis < dis:
                                dis = tempDis
                                mi = wi
                                mj = wj
                # newImg2[ti][tj] = dis^0.5 + newImg[ti][tj]
                newImg2[ti][tj] = 2 * newImg[mi][mj]

    return newImg2


def increase_width(train_imgs, train_groundTruth):
    weight=np.ones((train_imgs.shape[0],train_imgs.shape[1],train_imgs.shape[2]))*100
    # 测试
    for l in range(train_groundTruth.shape[0]):
        imgs = train_imgs[l]
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
                x1, y1 = y_edge[0][1], y_edge[0][0]
                x2, y2 = y_edge[1][1], y_edge[1][0]
                y_count=1000    ############################################################
                if y2 >= y1:
                    if np.sum(weight[l, y2 + 1:y2 + 4, x2]) == 300:
                        weight[l, y2 + 1, x2] = y_count
                        weight[l, y2 + 2, x2] = y_count
                        # weight[l, y2 + 3, x2] = y_count
                        # weight[l, y2 + 4, x2] = y_count
                    if np.sum(weight[l, y1 - 3:y1, x2]) == 300:
                        weight[l, y1 - 1, x1] = y_count
                        weight[l, y1 - 2, x1] = y_count
                        # weight[l, y1 - 3, x1] = y_count
                        # weight[l, y1 - 4, x1] = y_count
                else:
                    if np.sum(weight[l, y2 - 3:y2 , x2]) == 300:
                        weight[l, y2 - 1, x2] = y_count
                        weight[l, y2 - 2, x2] = y_count
                        # weight[l, y2 - 3, x2] = y_count
                        # weight[l, y2 - 4, x2] = y_count
                    if np.sum(weight[l, y1 + 1:y1 + 4, x2]) == 300:
                        weight[l, y1 + 1, x1] = y_count
                        weight[l, y1 + 2, x1] = y_count
                        # weight[l, y1 + 3, x1] = y_count
                        # weight[l, y1 + 4, x1] = y_count
            elif x_count<weight[l,x_point[0][0],x_point[0][1]]:
                for j in x_point:
                    if x_count<weight[l,j[0],j[1]]:
                        weight[l,j[0],j[1]]=x_count
                x1, y1 = x_edge[0][1], x_edge[0][0]
                x2, y2 = x_edge[1][1], x_edge[1][0]
                x_count=1000            ###########################################
                if x2 >= x1:
                    if np.sum(weight[l, y2, x2 +1: x2 +4]) == 300 :
                        weight[l, y2, x2+1] = x_count
                        weight[l, y2, x2+2] = x_count
                        # weight[l, y2, x2+3] = x_count
                        # weight[l, y2, x2+4] = x_count
                    if np.sum(weight[l, y2, x1 - 3: x1 ]) == 300:
                        weight[l, y1, x1-1] = x_count
                        weight[l, y1, x1-2] = x_count
                        # weight[l, y1, x1-3] = x_count
                        # weight[l, y1, x1-4] = x_count
                else:
                    if np.sum(weight[l, y2, x2 - 3: x2 ]) == 300 :
                        weight[l, y2, x2-1] = x_count
                        weight[l, y2, x2-2] = x_count
                        # weight[l, y2, x2-3] = x_count
                        # weight[l, y2, x2-4] = x_count
                    if np.sum(weight[l, y2, x1 + 1: x1 + 4]) == 300:
                        weight[l, y1, x1+1] = x_count
                        weight[l, y1, x1+2] = x_count
                        # weight[l, y1, x1+3] = x_count
                        # weight[l, y1, x1+4] = x_count
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
