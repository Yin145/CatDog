
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
#读取图片
from tensorflow.keras.preprocessing.image import array_to_img
im=cv2.imread(r'../imgs/cat.4.jpg')
im= cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = np.expand_dims(im, 0)

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

#设置生成器
datagen = ImageDataGenerator(
    #preprocessing_function=tf.image.per_image_standardization,
    preprocessing_function=zscore,
     #rescale=1./255.,
    channel_shift_range=100,
    #preprocessing_function=MeanFilter,
    rotation_range=10,  # 随机转动的最大角度
   )
datagen.fit(im)
#生成并画图
times=1
i = 0
for batch in datagen.flow(im, batch_size=1):
    plt.imshow(array_to_img(np.squeeze(batch)))
    print(batch)
    i += 1
    if i==times:
        plt.show()
        break