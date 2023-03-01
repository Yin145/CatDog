from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.losses import *
from utils.createModels import *
from utils.loaddata import load_data
from utils.plolfunction import *
import pandas as pd
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import *

# GPU按需分配声明
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''数据预处理'''
class_code = 'binary' #categorical
activiation = 'sigmoid'
num_classes = 1
input_shape = (64, 64, 3)
loss = binary_crossentropy
epochs =200
label_dict ={0: 'cat', 1: 'dog'}
#
train_gen = load_data(r'./datasets/traininfo.csv', class_code, train=True)
test_gen = load_data(r'./datasets/testinfo.csv', class_code)


'''定义训练参数'''
#设置步长和动态学习率
step_train = train_gen.n // train_gen.batch_size
step_test = test_gen.n // test_gen.batch_size

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

earlystop=EarlyStopping(monitor='acc',patience=5,min_delta=1e-3,verbose=1)

def train(model_name):
    if model_name=="vgg16":
        model = vgg16(input_shape, num_classes, activiation)
    elif model_name=='imp_vgg16':
        model = imp_vgg16(input_shape, num_classes, activiation)
    elif model_name=='resNet34':
        model=resNet34(input_shape,num_classes,activiation)
    else:
        return
    model.summary()
    model.compile(optimizer=Adam(lr=lr_schedule(0)),
                  loss=loss,
                  metrics=['acc'])
    model.fit_generator(train_gen, epochs=epochs, workers=8, steps_per_epoch=step_train,
                        shuffle=False,
                        callbacks=[TensorBoard(log_dir=f'./logs/{model_name}_logs'), lr_scheduler])
    test_loss, test_acc = model.evaluate_generator(test_gen, steps=step_test)
    print("loss:",test_loss, "acc",test_acc)
    test_gen.reset()
    preds = model.predict_generator(test_gen)

    if num_classes == 1:
        preds = np.where(preds > 0.5, 1, 0).reshape(-1).tolist()
    else:
        preds=preds.astype('int')
        preds=np.argmax(preds,axis=1).tolist()
    # acc=np.sum(preds==np.array(y_true))/len(y_true)
    print(classification_report(test_gen.classes, preds))
    conf = confusion_matrix(y_true=test_gen.classes, y_pred=preds)  # 混淆矩阵
    fpr, tpr, threshold = roc_curve(test_gen.classes, preds,drop_intermediate=False)   # 真正率和假正率
    roc_auc = auc(fpr, tpr)  # AUC分数
    print(f"AUC值:{roc_auc}\nFPR值:{fpr[1]}\nTPR值:{tpr[1]}")
    print("混淆矩阵如下：\n", conf)
    tn, fp, fn, tp = conf.ravel()
    print(tn,fp,fn,tp)

    #保存
    # if test_acc >= 0.87 and test_loss <= 0.4:
    #     model.save(f'./models/{model_name}.h5')
    # err_index = [i for i in range(len(true)) if true[i] != preds[i]]
    # print(err_index)
    # plot_wrong_pre(test_gen,true,preds,err_index)
    # import pickle
    # with open(f'{model_name}_fpr.pikle','wb')as f:
    #     pickle.dump(fpr,f)
    #     f.close()
    # with open(f'{model_name}_tpr.pikle','wb')as f:
    #     pickle.dump(tpr,f)
    #     f.close()
    # with open(f'{model_name}_roc_auc.pikle','wb')as f:
    #     pickle.dump(roc_auc,f)
    #     f.close()

train('resNet34')