from tensorflow.keras.layers import *
from tensorflow.keras import models
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

def Model1(input_shape, num_classes, activation):
    # keras的序贯模型
    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation=activation))
    return model


def Model2(input_shape, num_classes, activation):
    model = models.Sequential([
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation=activation)
    ])
    return model


def resNet34(input_shape, num_classes, activiation):  # (32, 32, 3)
    # 一个残差层
    def res_block(inputs, filters, strides, is_basic=False):
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same', kernel_initializer='he_normal'
                   ,kernel_regularizer=l2(1e-4)
                   )(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal'
                   ,kernel_regularizer=l2(1e-4)
                   )(x)
        x = BatchNormalization()(x)

        if is_basic:
            temp = Conv2D(filters=filters, kernel_size=(1, 1), strides=2, padding='same',
                          kernel_initializer='he_normal'
                          ,kernel_regularizer=l2(1e-4)
                          )(inputs)
            out = add([x, temp])
        else:
            out = add([x, inputs])
        out = Activation('relu')(out)
        return out

    inputs = Input(shape=input_shape)
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', strides=(1, 1),
               kernel_initializer='he_normal'   #正态分布初始化
               ,kernel_regularizer=l2(1e-4)
               )(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # layer2
    x = res_block(inputs=x, filters=16, strides=1)
    x = res_block(inputs=x, filters=16, strides=1)
    x = res_block(inputs=x, filters=16, strides=1)
    # layer3
    x = res_block(inputs=x, filters=32, strides=2, is_basic=True)
    x = res_block(inputs=x, filters=32, strides=1)
    x = res_block(inputs=x, filters=32, strides=1)
    x = res_block(inputs=x, filters=32, strides=1)
    # layer4
    x = res_block(inputs=x, filters=64, strides=2, is_basic=True)
    x = res_block(inputs=x, filters=64, strides=1)
    x = res_block(inputs=x, filters=64, strides=1)
    x = res_block(inputs=x, filters=64, strides=1)
    x = res_block(inputs=x, filters=64, strides=1)
    x = res_block(inputs=x, filters=64, strides=1)
    # layer5
    x = res_block(inputs=x, filters=128, strides=2, is_basic=True)
    x = res_block(inputs=x, filters=128, strides=1)
    x = res_block(inputs=x, filters=128, strides=1)
    x = GlobalAveragePooling2D()(x)
    #x = MaxPooling2D()(x)
    #x=AveragePooling2D()(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation=activiation, kernel_initializer='he_normal'
                    ,kernel_regularizer=l2(1e-4)
                    )(x)

    return Model(inputs, outputs)


def imp_vgg16(input_shape, num_classes, activiation):
    inputs = Input(shape=input_shape)
    # layer1
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               activation='relu'
               ,kernel_regularizer=l2(0.0005)
               )(inputs)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                        depth_multiplier=3)(x)
    x = Dropout(0.1)(x)
    # layer2
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2),
               activation='relu', padding='same'
               ,kernel_regularizer=l2(0.005)
               )(x)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Dropout(0.1)(x)

    # layer3
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               activation='relu', padding='same'
               ,kernel_regularizer=l2(0.005)
               )(x)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Dropout(0.4)(x)
    # layer4
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               activation='relu', padding='same'
               ,kernel_regularizer=l2(0.005)
              )(x)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    # layer5
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2),
               activation='relu', padding='same'
               ,kernel_regularizer=l2(0.005)
               )(x)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    # layer6
    x = Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2),
               activation='relu', padding='same'
               ,kernel_regularizer=l2(0.005)
               )(x)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)

    outputs = Dense(num_classes, activation=activiation)(x)
    return Model(inputs, outputs)



def vgg16(input_shape, num_classes, activiation):

    def conv_2t(x, filters):
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x=BatchNormalization()(x)
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
        return x

    def conv_3t(x, filters):
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
        return x

    filters=16
    inputs=Input(shape=input_shape)
    x=inputs
    for i in range(3):
        x=conv_2t(x,filters=filters)
        x=MaxPooling2D(pool_size=(2,2),padding='same',strides=(2,2))(x)
        filters=filters*2
    for i in range(2):
        x=conv_3t(x,filters=filters)
        x=MaxPooling2D(pool_size=(2,2),padding='same',strides=(2,2))(x)
    x=Flatten()(x)
    x=Dense(4096)(x)
    x=Activation('relu')(x)
    x=Dense(4096)(x)
    x=Activation('relu')(x)
    x=Dense(num_classes)(x)
    outputs=Activation(activiation)(x)
    return Model(inputs,outputs)

def Vgg16(input_shape,num_classes):
    from tensorflow.keras.applications import VGG16
    conv_base=VGG16(weights='imagenet',include_top=False,input_shape=input_shape,classes=num_classes)
    return conv_base