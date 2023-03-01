from tensorflow.keras.preprocessing.image import ImageDataGenerator
import  pandas as pd,numpy as np
import tensorflow as tf
def zscore(image):
    # zscores标准化
    image_zs = (image - np.mean(image)) / np.std(image)
    return image_zs
def MeanFilter(img, K_size=4):
    # 均值滤波
    h, w, c = img.shape
    # 零填充
    pad = K_size // 2
    out = np.zeros((h + 2 * pad, w + 2 * pad, c), dtype=np.float)
    out[pad:pad + h, pad:pad + w] = img.copy().astype(np.float)
    # 卷积的过程
    tmp = out.copy()
    for y in range(h):
        for x in range(w):
            for ci in range(c):
                out[pad + y, pad + x, ci] = np.mean(tmp[y:y + K_size, x:x + K_size, ci])

    out = out[pad:pad + h, pad:pad + w].astype(np.uint8)
    return out

def load_data(read_path,class_code,train=False):
    info=pd.read_csv(read_path)
    if train:
        info=info.sample(frac=1).reset_index()
        gen = ImageDataGenerator(
            #channel_shift_range=10,
            #rescale=1./255.,  # 缩放
             #rotation_range=10,#随机转动的最大角度
             #zoom_range=0.2,  # 随机缩放的最大幅度
            #height_shift_range=0.2,
            preprocessing_function=zscore

        )
        data_gen=gen.flow_from_dataframe(dataframe=info,x_col='image',y_col='label',
                                          target_size=(64,64),shuffle=True
                                         ,class_mode=class_code)

    else:
        gen=ImageDataGenerator(rescale=1./255.)
        data_gen=gen.flow_from_dataframe(dataframe=info,x_col='image',y_col='label',
                                          target_size=(64,64),shuffle=False
                                         , class_mode=class_code)
    return data_gen


#(32, 64, 64, 3) (32,) 输出64*64的RGB图像，每个批量包含32个样本

