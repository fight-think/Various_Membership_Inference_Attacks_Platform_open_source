#third package
import torch
import torch.optim as optim
import math
import sys
from datetime import datetime
import time
import pickle
import pandas as pd
import numpy as np
import os
import argparse
import yaml
import itertools
from copy import deepcopy
from random import sample

#self-defined package
from obtain_dataset import dataset_handler
from utils import get_device,relabel_mnist_shadow_data,detect_vulnerable_data_points
from model import model_mnist_CNN,model_mnist_ResNet18,model_cifar_LetNet,model_cifar_ResNet18,model_purchase_Shallow_MLP,model_purchase_Deeper_MLP
from train_process import train_one_dataset
from attack_feature_and_metric import obtain_feature_and_metric
from classifier_based_MIA import distinguish_with_classifier
from non_classifier_based_MIA import distinguish_without_classifier
from analyze_result import test_if_same,analyze_performance_result,analyze_infer_result,test_shape_of_infer_result,analyze_per_of_one_exp,analyze_infer_of_one_exp
from setting import cifar_args, mnist_args, purchase_args,cifar_100_args



parser = argparse.ArgumentParser("Input the path of config file")
parser.add_argument("config_path", type=str, default='./setting/mnist_set.yaml', help="The dataset with corresponding setting used for experiment in this excutation")
args = parser.parse_args()

with open(args.config_path, "r") as ymlfile:
     exe_args= yaml.load(ymlfile,Loader=yaml.FullLoader)


if exe_args!=None:

  save_prefix=""
  if "save_prefix" in exe_args.keys():
    save_prefix=exe_args['save_prefix']

  print(exe_args)
  target_dataset_name=exe_args['target_dataset_name'] #"PURCHASE-100"#"MNIST" CIFAR-10
  print(type(target_dataset_name))
  target_category_len=exe_args['target_category_len'] #the number of categories
  print(type(target_category_len))
  target_total_index_num=exe_args['target_total_index_num'] #400 #the number of total data samples in MNIST dataset 70000
  target_select_num=exe_args['target_select_num']  #400 #the number of data samples before each splitation 4000
  target_target_shadow_rate=exe_args['target_target_shadow_rate'] #0.5
  print(type(target_target_shadow_rate))
  target_target_train_rate=exe_args['target_target_train_rate']  #0.5
  target_shadow_train_rate=exe_args['target_shadow_train_rate'] #0.5
  target_split_num=exe_args['target_split_num'] #20 #the num of splitting for average the rate of being inferred
  #balance=0
  target_random_seed_for_dataset=exe_args['target_random_seed_for_dataset']
  target_fix_seed=exe_args['target_fix_seed']  #whether to use same seeds for reapting the experiments

  shadow_dataset_name=exe_args['shadow_dataset_name'] #"PURCHASE-100"#"MNIST" CIFAR-10
  print(type(shadow_dataset_name))
  shadow_category_len=exe_args['shadow_category_len'] #the number of categories
  print(type(shadow_category_len))
  shadow_total_index_num=exe_args['shadow_total_index_num'] #400 #the number of total data samples in MNIST dataset 70000
  shadow_select_num=exe_args['shadow_select_num']  #400 #the number of data samples before each splitation 4000
  shadow_target_shadow_rate=exe_args['shadow_target_shadow_rate'] #0.5
  print(type(shadow_target_shadow_rate))
  shadow_target_train_rate=exe_args['shadow_target_train_rate']  #0.5
  shadow_shadow_train_rate=exe_args['shadow_shadow_train_rate'] #0.5
  shadow_split_num=exe_args['shadow_split_num'] #20 #the num of splitting for average the rate of being inferred
  #balance=0
  shadow_random_seed_for_dataset=exe_args['shadow_random_seed_for_dataset']
  shadow_fix_seed=exe_args['shadow_fix_seed']  #whether to use same seeds for reapting the experiments

  hyper_paras=[]
  hyper_para_names=[]
  #define some other parameters
  target_metric=exe_args['target_metric']  #'accuracy'
  target_epoch_num=exe_args['target_epoch_num']  #100#2500 #100
  if type(target_epoch_num) is list:
    hyper_paras.append(deepcopy(exe_args['target_epoch_num']))
    hyper_para_names.append('target_epoch_num')
  target_batch_size=exe_args['target_batch_size'] #16 #36 #16
  if type(target_batch_size) is list:
    hyper_paras.append(deepcopy(exe_args['target_batch_size']))
    hyper_para_names.append('target_batch_size')
  target_learning_rate=exe_args['target_learning_rate'] #0.001 #0.001
  if type(target_learning_rate) is list:
    hyper_paras.append(deepcopy(exe_args['target_learning_rate']))
    hyper_para_names.append('target_learning_rate')
  target_model_name=exe_args['target_model_name'] #'model_mnist'#'model_purchase'#'model_mnist' model_cifar
  target_criterion=exe_args['target_criterion'] #"CrossEntropyLoss"
  target_weight_decay=exe_args['target_weight_decay'] #0.01
  if type(target_weight_decay) is list:
    hyper_paras.append(deepcopy(exe_args['target_weight_decay']))
    hyper_para_names.append('target_weight_decay')
  target_momentum=exe_args['target_momentum'] #0.9
  target_optimizer=exe_args['target_optimizer'] #"Adam" #Adam SGD
  if type(target_optimizer) is list:
    hyper_paras.append(deepcopy(exe_args['target_optimizer']))
    hyper_para_names.append('target_optimizer')
  target_model_select_strategy=exe_args['target_model_select_strategy'] #'highest_train_acc','highest_test_acc','highest_gap','lowest_gap'
  
  

  shadow_metric=exe_args['shadow_metric'] #'accuracy'
  shadow_epoch_num=exe_args['shadow_epoch_num'] #100#2500#1000 #100
  shadow_batch_size=exe_args['shadow_batch_size'] #16#64 #36 #16
  shadow_learning_rate=exe_args['shadow_learning_rate'] #0.001
  shadow_model_name=exe_args['shadow_model_name'] #'model_mnist'#'model_purchase'#'model_mnist'
  shadow_criterion=exe_args['shadow_criterion'] #"CrossEntropyLoss"
  shadow_weight_decay=exe_args['shadow_weight_decay'] #0.01
  shadow_momentum=exe_args['shadow_momentum'] #0.9
  shadow_optimizer=exe_args['shadow_optimizer'] #"Adam"
  shadow_model_select_strategy=exe_args['shadow_model_select_strategy']

  relabelled_shadow_metric=exe_args['relabelled_shadow_metric'] #'accuracy'
  relabelled_shadow_epoch_num=exe_args['relabelled_shadow_epoch_num'] #100#2500 #1000 #100
  relabelled_shadow_batch_size=exe_args['relabelled_shadow_batch_size'] #16#36 #64 #16
  relabelled_shadow_learning_rate=exe_args['relabelled_shadow_learning_rate'] #0.001
  relabelled_shadow_model_name=exe_args['relabelled_shadow_model_name'] #'model_mnist'#'model_purchase'#'model_mnist'
  relabelled_shadow_criterion=exe_args['relabelled_shadow_criterion'] #"CrossEntropyLoss"
  relabelled_shadow_weight_decay=exe_args['relabelled_shadow_weight_decay'] #0.01
  relabelled_shadow_momentum=exe_args['relabelled_shadow_momentum'] #0.9
  relabelled_shadow_optimizer=exe_args['relabelled_shadow_optimizer'] #"Adam"
  relabelled_shadow_model_select_strategy=exe_args['relabelled_shadow_model_select_strategy']

  #to load target model, shadow model, relabelled shadow model, and MIAs from previous experiments
  save_target_shadow_MIAs=False
  if "save_target_shadow_MIAs" in exe_args.keys():
    save_target_shadow_MIAs=exe_args['save_target_shadow_MIAs']
  
  pre_target_model="" #pre time stamp
  if "pre_target_model" in exe_args.keys():
    pre_target_model=exe_args['pre_target_model']
  
  pre_shadow_model=""
  if "pre_shadow_model" in exe_args.keys():
    pre_shadow_model=exe_args['pre_shadow_model']
  
  pre_MIAs=""
  if "pre_MIAs" in exe_args.keys():
    pre_MIAs=exe_args['pre_MIAs']

  #the strcuture used for storing information
  infer_result={}
  """
  {
    "MIA_1":[[[correctly_infer_time_while_as_member,,,],[correctly_infer_time_while_as_non-member,,,],[split_num 0-mem-predicted-as-non, 1-mem-mem, -1-not in],[split_num 0-non-mem-predicted-as-mem, 1-non-mem-non-mem, -1-not in]],,,]#length=total_index_num,
    "MIA_2":.....#initialize as [[[],[],[],[]]]*total_index_num if "MIA_i" not in the key list
  }
  """
  performance_result=[]
  """
  [{
    "target_model":[select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss]
    "shadow_model":[select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss]
    "shadow_model_with_relabelled_data":[select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss]
    "MIA_1":[TP,TN,FP,FN,accuracy,precision,recall],
    "MIA_2":.....
  },,,,,]
  """
  #2024/8/20 to store the attack models or thresholds for MIAs
  # classifier_based_MIA_path_or_threshold=[]
  # """
  # [{
  #   "MIA_1":model_1_path,
  #   "MIA_2":threshold_2,
  # }]
  # """

  #2024/8/20 give pre_target=pre_time_stamp or pre_shadow=pre_time_stamp or pre_MIA=pre_time_stamp
  #re-use previous target model or shadow model or MIA model/threshold

  now = datetime.now() # current date and time
  date_time = now.strftime("%Y-%d-%m-%H%M%S")
  if len(save_prefix)>0:
    dir_name=save_prefix+"/"+target_dataset_name+"-"+shadow_dataset_name+"-"+target_model_name+"-"+shadow_model_name+"-"+str(target_total_index_num)+"-"+str(target_select_num)+"-"+str(target_split_num)+"-"+str(target_fix_seed)+"-"+str(target_target_shadow_rate)+"-"+str(target_target_train_rate)+"-"+str(target_shadow_train_rate)+"-"+date_time
  else:
    dir_name=target_dataset_name+"-"+shadow_dataset_name+"-"+target_model_name+"-"+shadow_model_name+"-"+str(target_total_index_num)+"-"+str(target_select_num)+"-"+str(target_split_num)+"-"+str(target_fix_seed)+"-"+str(target_target_shadow_rate)+"-"+str(target_target_train_rate)+"-"+str(target_shadow_train_rate)+"-"+date_time
  start_time=time.time()

  model_load_save={}
  model_load_save['save_model_path']='./result/'+dir_name
  model_load_save['save_target_shadow_MIAs']=save_target_shadow_MIAs

  MIA_load_save={}
  MIA_load_save['save_model_path']='./result/'+dir_name
  MIA_load_save['pre_MIA_time_stamp']=pre_MIAs
  MIA_load_save['save_target_shadow_MIAs']=save_target_shadow_MIAs

  target_dataset=dataset_handler(target_dataset_name,target_total_index_num,target_select_num,target_target_shadow_rate,target_target_train_rate,target_shadow_train_rate)
  if target_dataset_name==shadow_dataset_name:
    shadow_dataset=target_dataset
  else:
    shadow_dataset=dataset_handler(shadow_dataset_name,shadow_total_index_num,shadow_select_num,shadow_target_shadow_rate,shadow_target_train_rate,shadow_shadow_train_rate)
  
  count_n=[] #[[member_time, non_member_time]...]
  for j in range(0,target_total_index_num):
    count_n.append([0,0])

  #privacy risk score
  prs_value=[]
  prs_index=[]
  prs_map={}
  #shapley value
  sv_value=[]
  sv_index=[]
  sv_map={}
  assert target_split_num==shadow_split_num

  valid_split_count=0 #only useful while evaluating mutiple settings of hyper-parameters
  valid_settings=[]

  #change the hyper-parameters of different target models, the hyper-parameters of the corresponding shadow and relabelled shadow models will also change
  #but you juse need to set a list of values for the settings of the target model 2023.7.8
  if len(hyper_paras)>0:
    para_product_list=list(itertools.product(*hyper_paras))
    print("np.shape(para_product_list):"+str(np.shape(para_product_list)))
    para_product_list=sample(para_product_list,len(para_product_list))
    if len(para_product_list)<target_split_num:
      raise Exception("The combination of the hyper-parameters is smaller than the split number")

    range_split_num=len(para_product_list)
  else:
    range_split_num=target_split_num

  for i in range(0,range_split_num): #consider parallelisation later
    current_time=time.time()
    
    MIA_count=0
    t_per_result={} #one dict in performance_result

    model_load_save['split_index']=i
    MIA_load_save['split_index']=i

    #set the hyper-parameters from the list of the product
    if len(hyper_para_names)>0:
      
      if valid_split_count==target_split_num: #find target_split_num models
        break
      
      if (valid_split_count+(range_split_num-i))<target_split_num: #can not find target_split_num models
        break

      print("set the hyper-parameters with:"+str(para_product_list[i]))
      for p_i in range(0,len(hyper_para_names)):
        if hyper_para_names[p_i]=='target_epoch_num':
          target_epoch_num,shadow_epoch_num,relabelled_shadow_epoch_num=deepcopy(para_product_list[i][p_i]),deepcopy(para_product_list[i][p_i]),deepcopy(para_product_list[i][p_i])
        elif hyper_para_names[p_i]=='target_batch_size':
          target_batch_size,shadow_batch_size,relabelled_shadow_batch_size=deepcopy(para_product_list[i][p_i]),deepcopy(para_product_list[i][p_i]),deepcopy(para_product_list[i][p_i])
        elif hyper_para_names[p_i]=='target_learning_rate':
          target_learning_rate,shadow_learning_rate,relabelled_shadow_learning_rate=deepcopy(para_product_list[i][p_i]),deepcopy(para_product_list[i][p_i]),deepcopy(para_product_list[i][p_i])
        elif hyper_para_names[p_i]=='target_weight_decay':
          target_weight_decay,shadow_weight_decay,relabelled_shadow_weight_decay=deepcopy(para_product_list[i][p_i]),deepcopy(para_product_list[i][p_i]),deepcopy(para_product_list[i][p_i])    
        elif hyper_para_names[p_i]=='target_optimizer':
          target_optimizer,shadow_optimizer,relabelled_shadow_optimizer=deepcopy(para_product_list[i][p_i]),deepcopy(para_product_list[i][p_i]),deepcopy(para_product_list[i][p_i])    
    
    #load and split dataset
    target_feature_c_name,target_label_c_name,target_target_train,target_target_train_index,target_target_test,target_target_test_index,target_shadow_train,target_shadow_train_index,target_shadow_test,target_shadow_test_index=target_dataset.split_dataset(i+target_random_seed_for_dataset,target_fix_seed)
    if target_dataset_name==shadow_dataset_name:
      shadow_feature_c_name,shadow_label_c_name,shadow_shadow_train,shadow_shadow_train_index,shadow_shadow_test,shadow_shadow_test_index=target_feature_c_name,target_label_c_name,target_shadow_train,target_shadow_train_index,target_shadow_test,target_shadow_test_index
    else:
      shadow_feature_c_name,shadow_label_c_name,_,_,_,_,shadow_shadow_train,shadow_shadow_train_index,shadow_shadow_test,shadow_shadow_test_index=shadow_dataset.split_dataset(i+shadow_random_seed_for_dataset,shadow_fix_seed)

    #count the member time and non-member time information
    for item in target_target_train_index:
      count_n[item][0]=count_n[item][0]+1
    for item in target_target_test_index:
      count_n[item][1]=count_n[item][1]+1
  
    device=get_device()

    print("***************************train target model*************************************")
    #train target model
    metric=target_metric
    epoch_num=target_epoch_num #paper 20000, try 1000 for test
    batch_size=target_batch_size  #paper 100, try 500 for test
    learning_rate=target_learning_rate
    weight_decay=target_weight_decay
    momentum=target_momentum
    if target_model_name in ['model_mnist_CNN','model_mnist_ResNet18','model_cifar_LetNet','model_cifar_ResNet18','model_purchase_Shallow_MLP','model_purchase_Deeper_MLP']:
      model=eval(target_model_name)().to(device)
    else:
      raise Exception(f"target_model_name--{target_model_name} not supported")

    if target_criterion=='CrossEntropyLoss':
      criterion=torch.nn.CrossEntropyLoss()
    if target_optimizer=='Adam':
      optimizer=optim.Adam(params=model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    elif target_optimizer=='SGD':
      optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum,weight_decay=weight_decay)
    elif target_optimizer=='RMSprop':
      optimizer = optim.RMSprop(model.parameters(),lr=learning_rate,momentum=momentum)

    model_load_save['pre_model_time_stamp']=pre_target_model
    model_load_save['target_shadow_or_reshadow']='target'
    target_model,select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss = train_one_dataset(target_target_train,target_target_test,epoch_num,batch_size,model,target_feature_c_name,target_label_c_name,device,optimizer,criterion,metric,target_model_name,target_dataset_name,target_model_select_strategy,model_load_save)
    
    #select the target model with a acc larger than 0.6 while training with mutiple settings of hyper-parameters
    if len(hyper_para_names)>0:
      if select_train_acc<0.6:
        continue
      else:
        valid_split_count+=1
        valid_settings.append(para_product_list[i])

    sub_dir_name="split-"+str(i)

    t_per_result['target_model']=[select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss]
    del select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss


    print("***************************train shadow model*************************************")
    #train shadow model (shadow_train and shadow_test)
    metric=shadow_metric
    epoch_num=shadow_epoch_num 
    batch_size=shadow_batch_size
    learning_rate=shadow_learning_rate
    weight_decay=shadow_weight_decay
    momentum=shadow_momentum
    
    if shadow_model_name in ['model_mnist_CNN','model_mnist_ResNet18','model_cifar_LetNet','model_cifar_ResNet18','model_purchase_Shallow_MLP','model_purchase_Deeper_MLP']:
      model=eval(shadow_model_name)().to(device)
    else:
      raise Exception(f"shadow_model_name--{shadow_model_name} not supported")
    
    if shadow_criterion=='CrossEntropyLoss':
      criterion=torch.nn.CrossEntropyLoss()
    if shadow_optimizer=='Adam':
      optimizer=optim.Adam(params=model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    elif shadow_optimizer=='SGD':
      optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum,weight_decay=weight_decay)
    elif shadow_optimizer=='RMSprop':
      optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum)
    
    model_load_save['pre_model_time_stamp']=pre_shadow_model
    model_load_save['target_shadow_or_reshadow']='shadow'
    shadow_model,select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss = train_one_dataset(shadow_shadow_train,shadow_shadow_test,epoch_num,batch_size,model,shadow_feature_c_name,shadow_label_c_name,device,optimizer,criterion,metric,shadow_model_name,shadow_dataset_name,shadow_model_select_strategy,model_load_save)
    t_per_result['shadow_model']=[select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss]
    del select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss

    #find vulnerable data points with privacy risk score and shapley value
    #1.privacy risk score
    #2.shapley value
    if target_dataset_name==shadow_dataset_name:
      try:
        value,index,value_1,index_1=detect_vulnerable_data_points(target_model_name,target_feature_c_name,target_target_train_index,target_target_train,target_target_test,shadow_shadow_train,shadow_shadow_test,device,target_model,shadow_model,target_category_len)
      except Exception:
        value,index,value_1,index_1=[],[],[],[]
        print("Error occur while obtaining vulnerable data points with privacy risk score and shapley value")
      prs_value.append(value)
      prs_index.append(index)
      sv_value.append(value_1)
      sv_index.append(index_1)
      num=int(target_total_index_num*0.01)

      print(index[0:num])
      print(value[0:num])

      for item in index[0:num]:
        if item not in prs_map.keys():
          prs_map[item]=1
        else:
          prs_map[item]=prs_map[item]+1

      for item in index_1[0:num]:
        if item not in sv_map.keys():
          sv_map[item]=1
        else:
          sv_map[item]=sv_map[item]+1
    
      print("***************************train shadow model with relabeled shadow data*************************************")
      #relabel shadow data with target model
      if pre_shadow_model=='':
        print("shadow_train.head()")
        print(shadow_shadow_train.head())
        relabeled_shadow_train,relabeled_shadow_test=relabel_mnist_shadow_data(target_model,shadow_shadow_train,shadow_shadow_test,shadow_feature_c_name,device,target_model_name,shadow_category_len)    
        print("shadow_train.head() after relabelling")
        print(shadow_shadow_train.head())
        print("relabeled_shadow_train['label'].value_counts()")
        print(relabeled_shadow_train['label'].value_counts())
      else:
        print("load from previous shadow model")
        relabeled_shadow_train,relabeled_shadow_test=shadow_shadow_train,shadow_shadow_test
      
      metric=relabelled_shadow_metric
      epoch_num=relabelled_shadow_epoch_num #paper 20000, try 1000 for test
      batch_size=relabelled_shadow_batch_size  #paper 100, try 500 for test
      learning_rate=relabelled_shadow_learning_rate
      weight_decay=relabelled_shadow_weight_decay
      momentum=relabelled_shadow_momentum
      if relabelled_shadow_model_name in ['model_mnist_CNN','model_mnist_ResNet18','model_cifar_LetNet','model_cifar_ResNet18','model_purchase_Shallow_MLP','model_purchase_Deeper_MLP']:
        model=eval(relabelled_shadow_model_name)().to(device)
      else:
        raise Exception(f"relabelled_shadow_model_name--{relabelled_shadow_model_name} not supported")
      
      if relabelled_shadow_criterion=='CrossEntropyLoss':
        criterion=torch.nn.CrossEntropyLoss()
      if relabelled_shadow_optimizer=='Adam':
        optimizer=optim.Adam(params=model.parameters(),lr=learning_rate,weight_decay=weight_decay)
      elif relabelled_shadow_optimizer=='SGD':
        optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum,weight_decay=weight_decay)
      elif relabelled_shadow_optimizer=='RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum)
      model_load_save['pre_model_time_stamp']=pre_shadow_model
      model_load_save['target_shadow_or_reshadow']='reshadow'
      relabel_shadow_model,select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss= train_one_dataset(relabeled_shadow_train,relabeled_shadow_test,epoch_num,batch_size,model,shadow_feature_c_name,shadow_label_c_name,device,optimizer,criterion,metric,relabelled_shadow_model_name,shadow_dataset_name,relabelled_shadow_model_select_strategy,model_load_save)
      t_per_result['shadow_model_with_relabelled_data']=[select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss]
      del select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss

    print("***************************get attack features and metric values******************************")
    min_p_float=sys.float_info.min

    e_criterion=torch.nn.CrossEntropyLoss(reduction='none') #the loss function of computing the loss metric
    basic_col_names,metric_col_names=[],[] #define the column names for different MIAs

    #constitute the attack dataset with shadow model and shadow dataset
    #iterate shadow dataset for getting the output of shadow model and compute some metrics
    express=pd.DataFrame()
    express,basic_col_names,metric_col_names=obtain_feature_and_metric(express,shadow_shadow_train,shadow_shadow_test,shadow_model,shadow_feature_c_name,e_criterion,min_p_float,device,shadow_model_name,shadow_category_len)
    del shadow_shadow_train
    del shadow_shadow_test

    #construct the test data for attack model based on the target_train and target_test data
    #iterate target dataset for getting the output of target model and compute some metrics
    express_for_target_data=pd.DataFrame()
    express_for_target_data,_,_=obtain_feature_and_metric(express_for_target_data,target_target_train,target_target_test,target_model,target_feature_c_name,e_criterion,min_p_float,device,target_model_name,target_category_len)
    
    #construct the attack feature based on relabeled shadow model
    #iterate relabeled_shadow_train dataset for getting the output of relabeled shadow model and compute some metrics
    express_for_relabeled_shadow_data=pd.DataFrame()
    express_for_re_target_data=pd.DataFrame()
    if target_dataset_name==shadow_dataset_name:
      
      express_for_relabeled_shadow_data,_,_=obtain_feature_and_metric(express_for_relabeled_shadow_data,relabeled_shadow_train,relabeled_shadow_test,relabel_shadow_model,shadow_feature_c_name,e_criterion,min_p_float,device,relabelled_shadow_model_name,shadow_category_len)
      del relabeled_shadow_train
      del relabeled_shadow_test

      #construct the target feature based on relabeled shadow model
      #iterate target dataset for getting the output of relabeled shadow model and compute some metrics
      express_for_re_target_data,_,_=obtain_feature_and_metric(express_for_re_target_data,target_target_train,target_target_test,relabel_shadow_model,target_feature_c_name,e_criterion,min_p_float,device,relabelled_shadow_model_name,target_category_len)
      del target_target_train
      del target_target_test


    print("***************************Classifier-based MIAs******************************")
    #the features used in different classifier-based MIAs

    print(express.shape)
    print(express_for_target_data.shape)
    print(express_for_relabeled_shadow_data.shape)
    print(express_for_re_target_data.shape)

    #basic_col_names,metric_col_names,rotate_col_names,shift_col_names --- four feature list returned by attack features and metrics computation
    if target_category_len==shadow_category_len:   
      #the shape of basi_col_names change from column names of confidence scores to three list, including confidence scores, logits, top-100 gradients
      conf_i=[]
      grad_i=[]
      logit_i=[]
      for i in range(len(basic_col_names)):
        if 'grad' not in basic_col_names[i] and 'logit' not in basic_col_names[i]:
          conf_i.append(basic_col_names[i])
        if 'grad' in basic_col_names[i]:
          grad_i.append(basic_col_names[i])
        if 'logit' in basic_col_names[i]:
          logit_i.append(basic_col_names[i])
      for i in range(len(metric_col_names)):
        print(f"metric--{i}:{metric_col_names[i]}")

      # conf_i=[0-num_classes-1]
      # grad_i=[top 100 gradients]
      # logit_i=[logit_0 to logit_num_classes-1]
      # metric_col_names=['max_p_value','loss','ground_truth_p','entropy','normalized_entropy','mentr','logit_gap','conf_gap']

      # classifier-based--full alignment of four different feature sources: C41: 4, C42: 6, C43: 4, C44:1, total 15， 15*4=60
      # threshold-based--len(metric_col_names)=8, 8*2+1=17, 1 means the gap_attack, *2 means relabel the label of the shadow data

      one_kind_features=[conf_i,grad_i,logit_i,metric_col_names]
      two_kind_features=[conf_i+grad_i, conf_i+logit_i, conf_i+metric_col_names, grad_i+logit_i, grad_i+metric_col_names, logit_i+metric_col_names]
      three_kind_features=[conf_i+grad_i+logit_i, conf_i+grad_i+metric_col_names, conf_i+logit_i+metric_col_names, grad_i+logit_i+metric_col_names]
      four_kind_features=[conf_i+grad_i+logit_i+metric_col_names]
      total_x_list=one_kind_features+two_kind_features+three_kind_features+four_kind_features
      #metric_features=['entropy','normalized_entropy','ground_truth_p','mentr','loss','max_p_value','logit_gap','conf_gap']
      #         one_kind_features   two_kind_features  three_kind_features  four_kind_features   gap_MIA   metric_features
      #features     (0~15)               (16~39)            (40~55)               (56~59)           60        61~61+8*2-1=76 

      #order of classifiers: SVM, LC, XGboost, NN
      #order of relabel: shadow, relabeled shadow
      print(f"after combination--len(total_x_list):{len(total_x_list)}")

    else: #the size of the output is not the same
      total_x_list=[metric_col_names]
      for item in metric_col_names:
        total_x_list.append([item])
    
    #some parameters used for MIAs:
    """
    MIA_count=0
    t_per_result={} #one dict in performance_result
    infer_result={}
    total_x_list #features list
    express #shadow dataset on shadow model   ex_0
    express_for_target_data #target dataset on target model ex_1
    express_for_relabeled_shadow_data #relabeled shadow dataset on relabeled shadow model ex_2
    express_for_re_target_data #target dataset on relabeled shadow model  ex_3
    """
    y_list=['label'] #the laber inf
    MIA_count,infer_result,t_per_result=distinguish_with_classifier(MIA_count,infer_result,t_per_result,total_x_list,y_list,express,express_for_target_data,device,target_total_index_num,MIA_load_save)
    test_shape_of_infer_result(infer_result,target_total_index_num)

    print("***************************non-classifier-based MIAs******************************")
    metric_names=['entropy','normalized_entropy','ground_truth_p','mentr','loss','max_p_value','logit_gap','conf_gap']
    MIA_count,infer_result,t_per_result=distinguish_without_classifier(metric_names,MIA_count,infer_result,t_per_result,express,express_for_target_data,express_for_relabeled_shadow_data,express_for_re_target_data,target_total_index_num,dir_name,sub_dir_name,target_category_len,MIA_load_save)
    test_shape_of_infer_result(infer_result,target_total_index_num)
    print("***************************analyze the attack performance and data samples' vulnerabilities******************************")
    #append performance
    performance_result.append(t_per_result)
    end_split_time=time.time()
    time_cost=int(end_split_time-current_time)
    
  
  #only save the result of last inference
  if len(hyper_paras)>0:
    if valid_split_count<target_split_num:
      raise Exception("Can not find enough target models satisfied with the requirement of training accuracy")
    else:
      print("The valid settings for the selected models:")
      print(str(valid_settings))

  dir_t_name="./result/"+dir_name+"/"+sub_dir_name+"/"
  if not os.path.exists(dir_t_name):
    os.makedirs(dir_t_name)

  #save_file infer_result
  result_path_1="./result/"+dir_name+"/"+sub_dir_name+"/infer_result_"+str(time_cost)+".txt"
  with open(result_path_1,'wb')as f:
    pickle.dump(infer_result,f)
  
  prs_list=[]
  sv_list=[]
  for key in prs_map.keys():
    prs_list.append([key,prs_map[key]])
  
  for key in sv_map.keys():
    sv_list.append([key,sv_map[key]])
  
  def fun_1(x):
    return x[1] #sort as test_avg
  prs_list.sort(key=fun_1,reverse=True)
  sv_list.sort(key=fun_1,reverse=True)
  prs_list=[item[0] for item in prs_list] #index as frequency
  sv_list=[item[0] for item in sv_list]
  num=int(target_total_index_num*0.01) #change to 0.01, get more vulnerable data points 2023.7.7
  
  prs_index_p="./result/"+dir_name+"/"+sub_dir_name+"/prs_index.txt"
  with open(prs_index_p,'w')as f:
    f.write(str(prs_index))
  prs_value_p="./result/"+dir_name+"/"+sub_dir_name+"/prs_value.txt"
  with open(prs_value_p,'w')as f:
    f.write(str(prs_value))
  del prs_value,prs_index

  prs_list_p="./result/"+dir_name+"/"+sub_dir_name+"/prs_list.txt"
  with open(prs_list_p,'w')as f:
    f.write(str(prs_list[0:num]))
  
  sv_index_p="./result/"+dir_name+"/"+sub_dir_name+"/sv_index.txt"
  with open(sv_index_p,'w')as f:
    f.write(str(sv_index))
  sv_value_p="./result/"+dir_name+"/"+sub_dir_name+"/sv_value.txt"
  with open(sv_value_p,'w')as f:
    f.write(str(sv_value))
  del sv_value,sv_index

  sv_list_p="./result/"+dir_name+"/"+sub_dir_name+"/sv_list.txt"
  with open(sv_list_p,'w')as f:
    f.write(str(sv_list[0:num]))
  
  #save_file performance_result
  end_time=time.time()
  time_cost=int(end_time-start_time)
  result_path_2="./result/"+dir_name+"/"+sub_dir_name+"/per_result_"+str(time_cost)+".txt"
  with open(result_path_2,'wb')as f:  #use w, we can read directly
    pickle.dump(performance_result,f)

  #save member and non-member information
  result_path_5="./result/"+dir_name+"/"+sub_dir_name+"/member_non_member_time.txt"
  with open(result_path_5,'wb')as f:
    pickle.dump(count_n,f)

  #analyze the result
  vul_as_test_avg,vul_as_train_avg=analyze_infer_result(infer_result,target_total_index_num,dir_name,sub_dir_name)
  result_path_3="./result/"+dir_name+"/"+sub_dir_name+"/"+"vul_as_test_avg.txt"
  with open(result_path_3,'w')as f:
    f.write(str(vul_as_test_avg))
  result_path_4="./result/"+dir_name+"/"+sub_dir_name+"/"+"vul_as_train_avg.txt"
  with open(result_path_4,'w')as f:
    f.write(str(vul_as_train_avg))
  analyze_performance_result(performance_result,dir_name,sub_dir_name)

  same_list=[]
  for item in vul_as_train_avg:
    if item in sv_list[0:num] and item in prs_list[0:num]:
      same_list.append(item)
  same_list_p="./result/"+dir_name+"/"+sub_dir_name+"/same_list.txt"
  with open(same_list_p,'w')as f:
    f.write(str(same_list))


  #test the analysis of specific MIA and target model
  train_avg_des,mem_t_as_train_avg,test_avg_des,non_mem_t_as_test_avg,train_test_des,mem_t_as_train_test,non_mem_t_as_train_test,final_dis_MIA_as_x=[],[],[],[],[],[],[],[]
  
  MIA_acc_of_datasets=[]
  table_list=[] #[[train_acc_avg,train_acc_var,test_acc_avg,test_acc_var,train_test_avg,train_test_var,MIA_acc_avg,MIA_acc_var,MIA_acc_med,MIA_acc_max,MIA_acc_min]]
  select_MIA_MIR_MT=[]
  select_MIA_NMIR_NMT=[]

  total_index_num=target_total_index_num
  infer_path=result_path_1
  with open(infer_path,'rb')as f:
    infer_result=pickle.load(f)
  train_avg_des,mem_t_as_train_avg,test_avg_des,non_mem_t_as_test_avg,train_test_des,mem_t_as_train_test,non_mem_t_as_train_test,final_dis_MIA_as_x,select_MIA_MIR_MT,select_MIA_NMIR_NMT,mem_accuracy_on_vulnerable,non_mem_accuracy_on_vulnerable=analyze_infer_of_one_exp(infer_result,total_index_num,train_avg_des,mem_t_as_train_avg,test_avg_des,non_mem_t_as_test_avg,train_test_des,mem_t_as_train_test,non_mem_t_as_train_test,final_dis_MIA_as_x,select_MIA_MIR_MT,select_MIA_NMIR_NMT,True)
  

  print("********mem_accuracy_on_vulnerable*******")
  print(repr(mem_accuracy_on_vulnerable.tolist()))

  mem_accuracy_on_vulnerable_path="./result/"+dir_name+"/"+sub_dir_name+"/mem_accuracy_on_vulnerable.txt"
  with open(mem_accuracy_on_vulnerable_path,'wb')as f:
    pickle.dump(mem_accuracy_on_vulnerable,f)

  print("********non_mem_accuracy_on_vulnerable*******")
  print(repr(non_mem_accuracy_on_vulnerable.tolist()))

  non_mem_accuracy_on_vulnerable_path="./result/"+dir_name+"/"+sub_dir_name+"/non_mem_accuracy_on_vulnerable.txt"
  with open(non_mem_accuracy_on_vulnerable_path,'wb')as f:
    pickle.dump(non_mem_accuracy_on_vulnerable,f)

  per_path=result_path_2
  with open(per_path,'rb')as f:
    performance_result=pickle.load(f)
  MIA_acc_of_datasets,table_list=analyze_per_of_one_exp(performance_result,MIA_acc_of_datasets,table_list)
 
