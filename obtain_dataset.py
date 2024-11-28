import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from copy import deepcopy
from random import sample

from utils import generate_md5

class dataset_handler:
  def __init__(self,dataset_name,total_index_num,select_num,target_shadow_rate,target_train_rate,shadow_train_rate):
    """
    total_index_num,int,data samples in specific distribution
    select_num,int,the total number of data samples in target and shadow dataset
    target_shadow_rate,float,the propotion of target dataset
    target_train_rate,float,the propotion of training data in target dataset
    shadow_train_rate,float,the propotion of training data in shadow dataset
    """
    self.dataset_name=dataset_name
    self.select_num=select_num
    self.total_index_num=total_index_num
    self.target_shadow_rate=target_shadow_rate
    self.target_train_rate=target_train_rate
    self.shadow_train_rate=shadow_train_rate

    if self.dataset_name=='MNIST':
      train_file_path='./dataset/mnist-in-csv/mnist_train.csv'
      mnist_train_data=pd.read_csv(train_file_path)
      feature_c_name=[item for item in mnist_train_data.columns if item != 'label']
      label_c_name=[]
      for i in range(0,10):
        label_c_name.append('label_'+str(i))
        mnist_train_data['label_'+str(i)]=0.0
        mnist_train_data.loc[mnist_train_data['label']==i,'label_'+str(i)]=1.0
      total_data=mnist_train_data.loc[[i for i in range(0,self.total_index_num)]]
      self.total_data=deepcopy(total_data)
      del total_data
      self.feature_c_name=feature_c_name
      self.label_c_name=label_c_name
    elif self.dataset_name=='CIFAR-10':
      file_path_train='./dataset/cifar-10-train-in-csv/train.csv'
      total_data=pd.read_csv(file_path_train)
      print("total_data.shape:"+str(total_data.shape))

      feature_c_name=[item for item in total_data.columns if item!='label']
      label_c_name=[]
      for i in range(0,10):
        label_c_name.append('label_'+str(i))
        total_data['label_'+str(i)]=0
        total_data.loc[total_data['label']==i,'label_'+str(i)]=1
      total_data[feature_c_name]=total_data[feature_c_name]/255.0
      total_data=total_data.loc[[i for i in range(0,self.total_index_num)]]
      self.total_data=deepcopy(total_data)
      del total_data
      self.feature_c_name=feature_c_name
      self.label_c_name=label_c_name
    elif self.dataset_name=='CIFAR-100':
      file_path='./dataset/cifar-100-train-in-csv/train.csv'
      total_data = pd.read_csv(file_path,index_col=[0])
      feature_c_name=[item for item in total_data.columns if item!='label']
      label_c_name=[]
      for i in range(0,100):
        label_c_name.append('label_'+str(i))
        total_data['label_'+str(i)]=0
        total_data.loc[total_data['label']==i,'label_'+str(i)]=1
      total_data[feature_c_name]=total_data[feature_c_name]/255.0
      total_data=total_data.loc[[i for i in range(0,self.total_index_num)]]
      self.total_data=deepcopy(total_data)
      del total_data
      self.feature_c_name=feature_c_name
      self.label_c_name=label_c_name
    elif self.dataset_name=='PURCHASE-100':
      file_path='./dataset/purchase-100-in-csv/purchase100.npz'
      data = np.load(file_path)
      features = data['features']
      labels = data['labels']
      feature_c_name=['feature_'+str(i) for i in range(0,600)]
      label_c_name=['label_'+str(i) for i in range(0,100)]
      t_1=pd.DataFrame(features, columns = feature_c_name, dtype='float32')
      t_2=pd.DataFrame(labels, columns = label_c_name)
      total_data=pd.concat([t_1,t_2],axis=1)
      for i in range(0,100):
        total_data.loc[total_data['label_'+str(i)]==1.0,'label']=i
      
      total_data=total_data.loc[[i for i in range(0,self.total_index_num)]]
      total_data['label']=total_data['label'].astype('int64')

      self.total_data=deepcopy(total_data)
      del total_data
      self.feature_c_name=feature_c_name
      self.label_c_name=label_c_name
    
  def split_dataset(self,random_num,fix_seed):
    if fix_seed:
      total_data=self.total_data.sample(n=self.select_num,random_state=random_num,replace = False)  
    else:
      total_data=self.total_data.sample(n=self.select_num,replace = False)

    #shuffle the dataset and divide them to four dataset  (mnist_train_data)
    total_index=[item for item in total_data.index]
    if fix_seed:
      target_train_test=total_data.sample(frac=self.target_shadow_rate,random_state=random_num,replace = False) #,random_state=split_num
    else:
      target_train_test=total_data.sample(frac=self.target_shadow_rate,replace = False) #,random_state=split_num
    target_index=[item for item in target_train_test.index]

    #check whether each split with different samples
    shadow_index=[item for item in total_index if item not in target_index]
    shadow_train_test=total_data.loc[shadow_index]

    ###target
    if fix_seed:
      target_train=target_train_test.sample(frac=self.target_train_rate,random_state=random_num,replace = False) #random_state=12,
    else:
      target_train=target_train_test.sample(frac=self.target_train_rate,replace = False) #random_state=12,
    target_train_index=[item for item in target_train.index]

    target_test_index=[item for item in target_index if item not in target_train_index]

    print(f"target_train_index_md5:{generate_md5(''.join([str(item) for item in target_train_index]))}")
    print(f"target_test_index_md5:{generate_md5(''.join([str(item) for item in target_test_index]))}")

    target_test=target_train_test.loc[target_test_index]

    ###shadow
    if fix_seed:
      shadow_train=shadow_train_test.sample(frac=self.shadow_train_rate,random_state=random_num,replace = False) #random_state=1,
    else:
      shadow_train=shadow_train_test.sample(frac=self.shadow_train_rate,replace = False) #random_state=1,
    shadow_train_index=[item for item in shadow_train.index]

    shadow_test_index=[item for item in shadow_index if item not in shadow_train_index]
    print(f"shadow_train_index_md5:{generate_md5(''.join([str(item) for item in shadow_train_index]))}")
    print(f"shadow_test_index_md5:{generate_md5(''.join([str(item) for item in shadow_test_index]))}")
    shadow_test=shadow_train_test.loc[shadow_test_index]

    return self.feature_c_name,self.label_c_name,target_train,target_train_index,target_test,target_test_index,shadow_train,shadow_train_index,shadow_test,shadow_test_index

      
