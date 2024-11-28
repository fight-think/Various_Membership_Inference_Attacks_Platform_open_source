import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from copy import deepcopy

from privacy_risk_score import black_box_benchmarks
from privacy_risk_score import calculate_risk_score
from shapley_value import KNN_Shapley #have to install Shapley with command: !python3 setup.py install

import hashlib


def generate_md5(input_str):
    
    assert type(input_str) == str, "the input should be a string"
    verify_str=hashlib.md5(input_str.encode()).hexdigest()
    return verify_str

#define method of finding vulnerable data points with privacy risk score and shapley value
def detect_vulnerable_data_points(target_model_name,feature_c_name,target_train_index,target_train,target_test,shadow_train,shadow_test,device,target_model,shadow_model,category_len):
  
  if 'model_mnist' in target_model_name:
    temp_train=target_train[feature_c_name].values.reshape(target_train.shape[0],1,28,28)
    temp_test=target_test[feature_c_name].values.reshape(target_test.shape[0],1,28,28)
    temp_1_train=shadow_train[feature_c_name].values.reshape(shadow_train.shape[0],1,28,28)
    temp_1_test=shadow_test[feature_c_name].values.reshape(shadow_test.shape[0],1,28,28)
  elif 'model_cifar' in target_model_name:
    temp_train=target_train[feature_c_name].values.reshape(target_train.shape[0],3,32,32)
    temp_test=target_test[feature_c_name].values.reshape(target_test.shape[0],3,32,32)
    temp_1_train=shadow_train[feature_c_name].values.reshape(shadow_train.shape[0],3,32,32)
    temp_1_test=shadow_test[feature_c_name].values.reshape(shadow_test.shape[0],3,32,32)
  elif 'model_cifar_100' in target_model_name:
    temp_train=target_train[feature_c_name].values.reshape(target_train.shape[0],3,32,32)
    temp_test=target_test[feature_c_name].values.reshape(target_test.shape[0],3,32,32)
    temp_1_train=shadow_train[feature_c_name].values.reshape(shadow_train.shape[0],3,32,32)
    temp_1_test=shadow_test[feature_c_name].values.reshape(shadow_test.shape[0],3,32,32)
  elif 'model_purchase' in target_model_name:
    temp_train=target_train[feature_c_name].values.reshape(target_train.shape[0],600)
    temp_test=target_test[feature_c_name].values.reshape(target_test.shape[0],600)
    temp_1_train=shadow_train[feature_c_name].values.reshape(shadow_train.shape[0],600)
    temp_1_test=shadow_test[feature_c_name].values.reshape(shadow_test.shape[0],600)
  feature_train=torch.from_numpy(temp_train).float().to(device)
  feature_test=torch.from_numpy(temp_test).float().to(device)
  feature_train_1=torch.from_numpy(temp_1_train).float().to(device)
  feature_test_1=torch.from_numpy(temp_1_test).float().to(device)
  
  target_model.eval()
  with torch.set_grad_enabled(False):
    y_train,_=target_model(feature_train)
    y_test,_=target_model(feature_test)
    y_train=y_train.cpu().numpy()
    y_test=y_test.cpu().numpy()
  
  shadow_model.eval()
  with torch.set_grad_enabled(False):
    y_train_1,_=shadow_model(feature_train_1)
    y_test_1,_=shadow_model(feature_test_1)
    y_train_1=y_train_1.cpu().numpy()
    y_test_1=y_test_1.cpu().numpy()
  print("shadow_train['label'].dtypes")
  print(shadow_train['label'].dtypes)
  print("shadow_test['label'].dtypes")
  print(shadow_test['label'].dtypes)
  shadow_train_performance=(y_train_1,shadow_train['label'].to_numpy())
  shadow_test_performance=(y_test_1,shadow_test['label'].to_numpy())
  print("target_train['label'].dtypes")
  print(target_train['label'].dtypes)
  print("target_test['label'].dtypes")
  print(target_test['label'].dtypes)
  target_train_performance=(y_train,target_train['label'].to_numpy())
  target_test_performance=(y_test,target_test['label'].to_numpy())
  num_classes=category_len

  #privacy risk score
  t_1=black_box_benchmarks(shadow_train_performance=shadow_train_performance,shadow_test_performance=shadow_test_performance,target_train_performance=target_train_performance,target_test_performance=target_test_performance,num_classes=num_classes)
  risk_score=calculate_risk_score(tr_values=t_1.s_tr_m_entr, te_values=t_1.s_te_m_entr, tr_labels=shadow_train['label'].to_numpy(), te_labels=shadow_test['label'].to_numpy(), data_values=t_1.t_tr_m_entr, data_labels=target_train['label'].to_numpy(), num_bins=5, log_bins=True)

  r_=[]
  for i in range(0,len(risk_score)):
    if risk_score[i]:
      r_.append([target_train_index[i],risk_score[i]])
  
  def fun_1(x):
    return x[1] #sort as test_avg
  r_.sort(key=fun_1,reverse=True)
  value=[item[1] for item in r_]
  index=[item[0] for item in r_]

  #shapley value
  measure = KNN_Shapley(K=5)
  res2 = measure._get_shapley_value_np(y_train, target_train['label'].to_numpy(), y_test, target_test['label'].to_numpy()).tolist()
  r_1=[]
  print("len(target_train_index)")
  print(len(target_train_index))
  print("len(res2")
  print(len(res2))
  for i in range(0,len(res2)):
    if res2[i]:
      r_1.append([target_train_index[i],res2[i]])
  r_1.sort(key=fun_1,reverse=True)
  value_1=[item[1] for item in r_1]
  index_1=[item[0] for item in r_1]

  return value,index,value_1,index_1

#the dataset defined for speeding the training of models
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, f_s,l_s,dataset_name):
    'Initialization'
    self.f_s=f_s
    self.l_s=l_s
    self.dataset_name=dataset_name


  def __len__(self):
    'Denotes the total number of samples'
    return np.shape(self.f_s)[0]

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    if self.dataset_name=='MNIST':
      x=torch.from_numpy(self.f_s[index].reshape(1,28,28)).float()
    elif self.dataset_name=='PURCHASE-100':
      x=torch.from_numpy(self.f_s[index]).float()
    elif self.dataset_name=='CIFAR-10':
      x=torch.from_numpy(self.f_s[index].reshape(3,32,32)).float()
    elif self.dataset_name=='CIFAR-100':
      x=torch.from_numpy(self.f_s[index].reshape(3,32,32)).float()
    y=torch.from_numpy(self.l_s[index]).float()

    return x,y


#get the device in the environment
def get_device():
  print("*******device information*******")
  if torch.cuda.is_available():
    print("GPU:")
    device = torch.device('cuda:0')
    print(torch.cuda.get_device_properties(device))

  else:
    print("Only CPU")
    device = torch.device('cpu') # don't have GPU 
  return device


#the function of computing evalution metric
def report_metric(model_output,y,metric):
  """
  model_output, tensor, the output after the deal of training model
  y, tensor, the label for data,
  metric,str, str, the evaluation metric
  """
  if metric=='accuracy':
    count=0
    rows=model_output.shape[0]
    for i,item in enumerate(model_output):
        
      temp1=model_output[i].cpu().detach().numpy()
      temp2=y[i].cpu().numpy()
      
      if np.argmax(temp1)==np.argmax(temp2):
        count+=1
    return count/rows
  elif metric=='binary_accuracy':
    count=0
    rows=model_output.shape[0]
    for i,item in enumerate(model_output):
        
      temp1=model_output[i].cpu().detach().numpy()
      temp2=y[i].cpu().numpy()
      
      #binary if prediction_value>0.5 label 1 else label 0
      if (temp1[0]>0.5 and temp2[0]==1) or (temp1[0]<=0.5 and temp2[0]==0) :
        count+=1
    return count/rows

def relabel_mnist_shadow_data(target_model,shadow_train_t,shadow_test_t,feature_c_name,device,model_name,category_len):
  """
  target_model,nn.model,trained target model
  shadow_train_t,pd.DataFrame,the dataset used for training shadow model
  shadow_test_t,pd.DataFrame,the dataset used for testing shadow model
  feature_c_name,list,the columns' names in DataFrame for the feature
  category_len,int,the number of categories in this dataset
  """
  #construct relabel shadow dataset based on target model
  shadow_train=deepcopy(shadow_train_t)
  shadow_test=deepcopy(shadow_test_t)
  shadow_train['original_label']=shadow_train['label']
  shadow_test['original_label']=shadow_test['label']

  if 'model_mnist' in model_name:
    temp_train=shadow_train[feature_c_name].values.reshape(shadow_train.shape[0],1,28,28)
    temp_test=shadow_test[feature_c_name].values.reshape(shadow_test.shape[0],1,28,28)
  elif 'model_cifar' in model_name:
    temp_train=shadow_train[feature_c_name].values.reshape(shadow_train.shape[0],3,32,32)
    temp_test=shadow_test[feature_c_name].values.reshape(shadow_test.shape[0],3,32,32)
  elif 'model_cifar_100' in model_name:
    temp_train=shadow_train[feature_c_name].values.reshape(shadow_train.shape[0],3,32,32)
    temp_test=shadow_test[feature_c_name].values.reshape(shadow_test.shape[0],3,32,32)
  elif 'model_purchase' in model_name:
    temp_train=shadow_train[feature_c_name].values.reshape(shadow_train.shape[0],600)
    temp_test=shadow_test[feature_c_name].values.reshape(shadow_test.shape[0],600)
  feature_train=torch.from_numpy(temp_train).float().to(device)
  feature_test=torch.from_numpy(temp_test).float().to(device)
  target_model.eval()
  with torch.set_grad_enabled(False):
    y_train,_=target_model(feature_train)
    y_test,_=target_model(feature_test)

  t_train=torch.argmax(y_train,dim=1).cpu().numpy()
  t_test=torch.argmax(y_test,dim=1).cpu().numpy()

  shadow_train['label']=t_train
  shadow_test['label']=t_test
  print("Relabelled shadow train total:%d; changed:%d"%(shadow_train.shape[0],shadow_train[shadow_train['label']==shadow_train['original_label']].shape[0]))
  print("Relabelled shadow test total:%d; changed:%d"%(shadow_test.shape[0],shadow_test[shadow_test['label']==shadow_test['original_label']].shape[0]))
  
  for i in range(0,category_len):
    shadow_train.loc[shadow_train['label']==i,'label_'+str(i)]=1.0
    shadow_train.loc[shadow_train['original_label']==i,'label_'+str(i)]=0.0
    
    shadow_test.loc[shadow_test['label']==i,'label_'+str(i)]=1.0
    shadow_test.loc[shadow_test['original_label']==i,'label_'+str(i)]=0.0
  
  return shadow_train, shadow_test

#X_train, X_test, y_train, y_test = train_test_split(express.loc[ : ,x_list], express['label'], test_size=0.20, random_state=1231)
def spilt_attack_train_and_test(data, test_size, random_seed,x_list,y_list):
  """
  data, DataFrame, the data used for training attack model
  test_size, Float, the percentage of test dataset
  random_seed, int, the random seed of selecting data
  x_list, list, the feature names
  y_list, list, the label names
  """
  all_index=[item for item in data.index]
  test_dataset=data.sample(frac=test_size,random_state=random_seed,replace = False)
  test_index=[item for item in test_dataset.index]
  
  train_index=[item for item in all_index if item not in test_index]
  train_dataset=data.loc[train_index]
  X_train=train_dataset[x_list]
  y_train=train_dataset[y_list]
  
  X_test=test_dataset[x_list]
  y_test=test_dataset[y_list]
    
  return train_dataset,test_dataset,X_train, X_test, y_train, y_test


#define the common function of drawing a picture
def draw_a_pic(line_labels,dis_data,x_tick_num,x_ticks,pic_name,dir_name,sub_dir_name,x_label,y_label,pic_type):
  #parameters:
  """
  line_labels,list of string,labels of multiple lines
  dis_data,list of list,[[value of label 1, label 2,,,,],[]] 
  x_tick_num,int,the number of numbers displayed in x-axis.If specified, x_ticks from 0 to len(dis_data)-1
  x_ticks,list,if specified, use this rather than x_tick_num
  pic_name,str,the name of saved picture
  dir_name,str,dir name of saving the picture of different split
  sub_dir_name,str,dir name of saving the picture for each split
  pic_type,str,'1_1' one row and one column; '2_1' two rows and one column;
  """

  #draw dis_Index_as_x, dis_MIA_as_x
  len_x=len(dis_data)
  x=[i for i in range(0,len_x)]
  if len(x_ticks)==0:
    step=int((len_x-1)/(x_tick_num-1))
    x_ticks=[0+i*step for i in range(1,x_tick_num-1)]
    x_ticks.append(len_x-1)
    x_ticks=[0]+x_ticks
  if pic_type=='1_1':
    for j in range(0,len(line_labels)):
      plt.plot(x, [dis_data[i][j] for i in range(0,len_x)], label = line_labels[j])
    plt.xticks(x_ticks)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(pic_name)

    
  elif pic_type=='2_1':
    fig, axs = plt.subplots(2, 1, sharex=True) 
    axs[0].plot(x, [dis_data[i][0] for i in range(0,len_x)], label = line_labels[0],color='tab:blue')
    axs[0].plot(x, [dis_data[i][3] for i in range(0,len_x)], label = line_labels[3],color='tab:orange')
    axs[0].legend() 
    axs[0].set_title(pic_name)
    axs[0].set_ylabel(y_label)

    axs[1].plot(x, [dis_data[i][1] for i in range(0,len_x)], label = line_labels[1],color='tab:green')
    axs[1].plot(x, [dis_data[i][2] for i in range(0,len_x)], label = line_labels[2],color='tab:red')
    axs[1].plot(x, [dis_data[i][4] for i in range(0,len_x)], label = line_labels[4],color='tab:purple')
    axs[1].plot(x, [dis_data[i][5] for i in range(0,len_x)], label = line_labels[5],color='tab:brown')
    axs[1].legend() #bbox_to_anchor=(1.24,1), loc="upper right"
    

    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(y_label)
    plt.xticks(x_ticks)
  
  elif pic_type=='3_1':
    fig, axs = plt.subplots(3, 1, sharex=True) #, tight_layout=True
    axs[0].plot(x, [dis_data[i][0] for i in range(0,len_x)], label = line_labels[0],color='tab:blue')
    axs[0].plot(x, [dis_data[i][1] for i in range(0,len_x)], label = line_labels[1],color='tab:orange')
    axs[0].set_title(pic_name)
    axs[0].legend() #loc="upper right" bbox_to_anchor=(1.5,1), loc="upper right"
    axs[0].set_ylabel(y_label)

    axs[1].plot(x, [dis_data[i][2] for i in range(0,len_x)], label = line_labels[2],color='tab:green')
    axs[1].plot(x, [dis_data[i][3] for i in range(0,len_x)], label = line_labels[3],color='tab:red')
    axs[1].legend() #bbox_to_anchor=(1.24,1), loc="upper right"
    axs[1].set_ylabel(y_label)

    axs[2].plot(x, [dis_data[i][4] for i in range(0,len_x)], label = line_labels[4],color='tab:purple')
    axs[2].plot(x, [dis_data[i][5] for i in range(0,len_x)], label = line_labels[5],color='tab:brown')
    axs[2].legend()
    
    axs[2].set_xlabel(x_label)
    axs[2].set_ylabel(y_label)
    plt.xticks(x_ticks)

  pic_name=pic_name+".png"
  dir_t_name="./result/"+dir_name+"/"+sub_dir_name+"/"
  if not os.path.exists(dir_t_name):
    os.makedirs(dir_t_name)
  plt.savefig(os.path.join(dir_t_name,pic_name), dpi=150)
  plt.clf()