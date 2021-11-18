#https://github.com/MrGiovanni/UNetPlusPlus
import keras
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose,core,Add,Multiply,Concatenate
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.noise import GaussianDropout
import h5py
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils import to_categorical
from PIL import Image
from keras.utils import multi_gpu_model
from sklearn.metrics import roc_auc_score
from keras.callbacks import TensorBoard
import os
# 原始参数
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


"""
Standard UNet++ [Zhou et.al, 2018]
Total params: 9,041,601
"""
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print('GPU', tf.test.is_gpu_available())

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)


def Nest_Net(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=True):
    ########################################
    # 2D Standard
    ########################################
    smooth = 1.
    dropout_rate = 0.2

    def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):
        x = Conv2D(nb_filter, (kernel_size, kernel_size), activation="relu", name='conv' + stage + '_1',
                   kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4),
                   data_format='channels_first')(input_tensor)
        x = Dropout(dropout_rate, name='dp' + stage + '_1')(x)
        x = Conv2D(nb_filter, (kernel_size, kernel_size), activation="relu", name='conv' + stage + '_2',
                   kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4),
                   data_format='channels_first')(x)
        x = Dropout(dropout_rate, name='dp' + stage + '_2')(x)

        return x

    ########################################

    global bn_axis

    bn_axis = 1
    img_input_1 = Input(shape=(color_type, img_rows, img_cols), name='img_input_1')

    nb_filter = [32, 64, 128, 256, 512]

    conv1_2 = standard_unit(img_input_1, stage='12', nb_filter=nb_filter[0])
    pool1_2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1_2', data_format='channels_first')(conv1_2)

    # conv2_2 = standard_unit(pool1_2, stage='22', nb_filter=nb_filter[1])
    conv2_2 = res2net_bottleneck_block(pool1_2, nb_filter[0], nb_filter[1],s=int(nb_filter[0]/4.0))
    pool2_2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2_2', data_format='channels_first')(conv2_2)

    conv3_2 = res2net_bottleneck_block(pool2_2, nb_filter[1], nb_filter[2],s=int(nb_filter[1]/4.0))
    # conv3_2 = standard_unit(pool2_2, stage='32', nb_filter=nb_filter[2])

    up8_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up82', padding='same',
                            data_format='channels_first')(conv3_2)
    conv8_2 = concatenate([up8_2, conv2_2], name='merge82', axis=bn_axis)
    # conv8_2 = standard_unit(conv8_2, stage='82', nb_filter=nb_filter[1])
    conv8_2 = res2net_bottleneck_block(conv8_2, nb_filter[2], nb_filter[1],s=int(nb_filter[2]/4.0))

    up9_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up92', padding='same',
                            data_format='channels_first')(conv8_2)
    conv9_2 = concatenate([up9_2, conv1_2], name='merge92', axis=bn_axis)
    # conv9_2 = standard_unit(conv9_2, stage='92', nb_filter=nb_filter[0])
    conv9_2 = res2net_bottleneck_block(conv9_2, nb_filter[1], nb_filter[0],s=int(nb_filter[1]/4.0))

    reshape_1 = Conv2D(2, (1, 1), activation='relu', name='relu', kernel_initializer='he_normal', padding='same',
                       kernel_regularizer=l2(1e-4), data_format='channels_first')(conv9_2)
    reshape_1 = core.Reshape((2, patch_h * patch_w), name='output1')(reshape_1)
    reshape_1 = core.Permute((2, 1))(reshape_1)
    reshape_1 = core.Activation('softmax', name='reshape_1')(reshape_1)

    model = Model(inputs=img_input_1, outputs=[reshape_1])

    # model = multi_gpu_model(model, gpus=2)  # 就插入到这里

    model.compile(optimizer=Adam(lr=3e-4), loss={'reshape_1':"categorical_crossentropy"},
                  metrics=['accuracy'])

    return model


def res2net_bottleneck_block(x, f, out_channels, s=4, use_se_block=True):
    """
    Arguments:
        x: input tensor
        f: number of input  channels
        s: scale dimension
    """
    # 获取输入特征的通道
    num_channels = int(x._keras_shape[-3])
    assert (num_channels == f)
    input_tensor = x
    # Conv 1x1
    if f==1:
        x = Conv2D(out_channels, 1, kernel_initializer='he_normal', use_bias=False, data_format='channels_first')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        f=out_channels
        s=int(out_channels/4.0)
    else:
        x = Conv2D(f, 1, kernel_initializer='he_normal', use_bias=False, data_format='channels_first')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # Conv 3x3
    # 定义一个数组来存放 y1,y2,y3,y4模块
    subset_x = []

    n = f
    # w表示将通道分成几组后，每组通道数
    w = n // s
    mx = Lambda(lambda x: tf.split(x, s, axis=1))(x)
    # 分几组循环几次
    for i in range(s):
        slice_x = mx[i]
        # 当第0组是时，x1不计算，直接保存特征到subset_x，
        # 当第1组时，x2经过3X3卷积后产生K2,保存到subset_x
        # 当第2组时，x3和从subset_x取出的K2,进行融合，再经过3X3卷积后产生K3,保存到subset_x
        # 当第3组时，同上步
        if i > 1:
            slice_x = Add()([slice_x, subset_x[-1]])
        if i > 0:
            slice_x = Conv2D(w, 3, kernel_initializer='he_normal', padding='same', use_bias=False,
                             data_format='channels_first')(slice_x)
            slice_x = BatchNormalization()(slice_x)
            slice_x = Activation('relu')(slice_x)
        subset_x.append(slice_x)
    # 将subset_x中保存的 y1，y2, y3, y4 经行 concat
    x = Concatenate(axis=1)(subset_x)

    # Conv 1x1
    x = Conv2D(out_channels, 1, kernel_initializer='he_normal', use_bias=False, data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 下面是res2net结合senet块，可用可不用
    if use_se_block:
        x = core.Permute((2,3,1))(x)
        x = se_block(x)
        x = core.Permute((3,1,2))(x)

    # skip = input_tensor
    # Add
    if num_channels == out_channels:
        skip = input_tensor
    else:
        skip = input_tensor
        skip = Conv2D(out_channels, 1, kernel_initializer='he_normal', data_format='channels_first')(skip)
    out = Add()([x, skip])
    return out


def se_block(input_tensor, c=16):
    num_channels = int(input_tensor._keras_shape[-1])  # Tensorflow backend
    bottleneck = int(num_channels // c)

    se_branch = GlobalAveragePooling2D()(input_tensor)
    se_branch = Dense(bottleneck, use_bias=False, activation='relu')(se_branch)
    se_branch = Dropout(0.2)(se_branch)
    se_branch = Dense(num_channels, use_bias=False, activation='sigmoid')(se_branch)
    se_branch = Dropout(0.2)(se_branch)

    out = Multiply()([input_tensor, se_branch])
    return out


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


# 下载训练数据
print("正在下载训练数据...")
train_imgs = load_hdf5("./u_net_data/DRIVE_dataset_imgs_train.hdf5")
train_groundTruth = load_hdf5("./u_net_data/DRIVE_dataset_groundTruth_train.hdf5")
print("训练数据下载完成...")

# 可视化训练集和标签
dex=np.random.randint(0,train_imgs.shape[0],size=50)
visiable_imgs=np.copy(train_imgs)
visiable_imgs=visiable_imgs.reshape((train_imgs.shape[0],train_imgs.shape[2],train_imgs.shape[3]))
visiable_label=np.copy(train_groundTruth)
visiable_label=visiable_label.reshape((train_imgs.shape[0],train_imgs.shape[2],train_imgs.shape[3]))
Picture_1=visiable_imgs[dex[0]]
Picture_2=visiable_label[dex[0]]
for i in dex:
    Picture_1 = np.concatenate((Picture_1, visiable_imgs[i]), axis=1)
    Picture_2 = np.concatenate((Picture_2, visiable_label[i]), axis=1)
full_imgs=np.concatenate((Picture_1, Picture_2), axis=0)
full_imgs=np.uint8(full_imgs*255)
prediction = Image.fromarray(full_imgs, mode="L")
prediction.save("./u_net_data/"+"visiable_training_imgs_and_label.png")

# 编译模型
model = Nest_Net(patch_h, patch_w)

json_string = model.to_json()
open('./RES2NET/architecture.json', 'w').write(json_string)


# 学习率下降
def step_decay(epoch):
    print("学习率为:", K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)


print_lrate = LearningRateScheduler(step_decay)
lrate=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, mode='auto', cooldown=0, min_lr=3e-7)
model_save = keras.callbacks.ModelCheckpoint(filepath='./RES2NET/my_model.h5', verbose=0)
best_weight = ModelCheckpoint(filepath='./RES2NET/best_weight.h5', verbose=1, monitor='val_loss',mode='auto', save_best_only=True)

# 训练模型
assert (np.min(train_groundTruth) == 0 and (np.max(train_groundTruth) == 1))
train_groundTruth = train_groundTruth.reshape(per_img_patch_count * Nimgs, patch_h * patch_w)
train_groundTruth = to_categorical(train_groundTruth)

model.fit(x=train_imgs,y={'reshape_1':train_groundTruth},
          batch_size=32, epochs=100, verbose=1, shuffle=True,validation_split=0.1,
          callbacks=[lrate,print_lrate,model_save,best_weight])

model.save('./RES2NET/last_weight.h5', overwrite=True)
