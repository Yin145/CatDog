import os
import pandas as pd

def dataio(open_path,save_path):
    images=[]
    labels=[]
    for root,_,filenames in os.walk(open_path):
        root=root[1:len(root)]
        for filename in filenames:
            image_path=os.path.join(root,filename)
            images.append(image_path)
            s=filename[0:3]
            labels.append(s)
    dataframe=pd.DataFrame({'image':images,"label":labels})
    dataframe.to_csv(save_path,index=False)

dataio('../datasets/test','../datasets/testinfo.csv')
dataio('../datasets/train','../datasets/traininfo.csv')