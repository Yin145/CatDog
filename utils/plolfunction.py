import matplotlib.pyplot as plt
import numpy as np,pandas as pd
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def plot_matrix(fpr,tpr,roc_auc):  #labels=gen.class_indices
    lw = 1
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='VGG (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot(fpr_tf, tpr_tf, color='red',
    #          lw=lw, label='TF-IDF-SVM (area = %0.2f)' % roc_auc_tf)  ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot(fpr_ls, tpr_ls, color='blue',
    #          lw=lw, label='LSTM (area = %0.2f)' % roc_auc_ls)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def plot_aug(gen,label_dict,rows,cols):
    fig=plt.figure()
    x_batch,y_batch=gen[0]  #x_batch[i].shape:(64, 64, 3)
    print(y_batch)
    for i in range(0,(rows*cols)):
        ax=fig.add_subplot(rows,cols,i+1)
        ax.imshow(np.squeeze(x_batch[i]))
        ax.set_title(str(i)+" "+label_dict[y_batch[i]])
    fig.suptitle('Augmentd Images')
    plt.show()



