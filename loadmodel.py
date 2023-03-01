from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.metrics import *
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

#从路径读取测试集
def get_img():
    imgs_path=info['image'].apply(lambda x:x.replace('\\',"/")).tolist()
    imgs_tensor=[]
    imgs=[]
    for img_path in imgs_path:
        img = image.load_img(img_path, target_size=(64, 64))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        imgs_tensor .append(img_tensor)
        imgs.append(img)
    imgs_tensor=np.array(imgs_tensor).reshape(1000,64,64,3)
    return imgs_tensor,imgs

#从生成器读取测试
def get_gen():
    test_gen = ImageDataGenerator(rescale=1. / 255.).flow_from_dataframe(dataframe=info, x_col='image', y_col='label',
                                                                         target_size=(64, 64), class_mode='binary',
                                                                         batch_size=32,
                                                                         shuffle=False)
    return test_gen

def evaluate(labels,gen=None,img=None):
    if gen!=None:
        preds = model.predict_generator(gen)
        test_loss, test_acc = model.evaluate_generator(gen)
    if img.all()!=None:
        preds = model.predict(img)
        test_loss, test_acc=model.evaluate(imgs_tensor,labels)

    print(f"test_loss:{test_loss},test_acc:{test_acc}")
    preds = np.where(preds > 0.5, 1, 0).reshape(-1)
    acc = np.sum(preds == labels) / len(labels)
    err_index = [i for i in range(len(labels)) if labels[i] != preds[i]]
    labels = labels.tolist()
    preds=preds.tolist()
    conf = confusion_matrix(y_true=labels, y_pred=preds)

    fpr, tpr, threshold = roc_curve(labels, preds,drop_intermediate=False)   # 真正率和假正率
    roc_auc = auc(fpr, tpr)  # AUC分数
    print(f'fpr:{fpr[1]} tpr:{tpr[1]} roc_auc:{roc_auc}\n混淆矩阵:\n{conf}')
    print(f"acc:{acc}")
    plt.plot(fpr,tpr,label='Roc')
    plt.show()
    return preds,err_index,fpr,tpr,roc_auc

#读取单张图像进行预测
def predict(path):
    img = image.load_img(path, target_size=(64, 64))
    img_tensor = image.img_to_array(img)/255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    #plt.imshow(img_tensor[0]) #img
    #plt.show()
    pred=model.predict(img_tensor)
    if pred>=1:
        print(f"{path} : is a dog.")
    else:
        print(f"{path} : is a cat.")




#查看预测错误图像
def plot_wrong_img(preds,err_index):
    fig=plt.figure()
    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(imgs[err_index[i]])
        img_index=err_index[i]
        ax.set_title(f'{label_dict[labels.tolist()[img_index]]}/{label_dict[preds[img_index]]}')
    plt.show()

if __name__ == '__main__':
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    path = "imp_vgg16.h5"
    label_dict = {0: 'cat', 1: 'dog'}
    info = pd.read_csv('./datasets/testinfo.csv', )
    model = load_model('./models/' + path)
    labels=info['label'].apply(lambda x:0 if x=='cat' else 1).values #np.array

    imgs_tensor,imgs=get_img()
    preds,err_index,fpr,tpr,roc_auc=evaluate(labels,img=imgs_tensor)
    #predict(r'./datasets/test/cat.10002.jpg')
    plot_wrong_img(preds,err_index)
