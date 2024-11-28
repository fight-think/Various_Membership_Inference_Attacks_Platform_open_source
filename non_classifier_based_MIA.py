import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,roc_curve
import matplotlib.pyplot as plt
import os
import json

def distinguish_without_classifier(metric_names,MIA_count,infer_result,t_per_result,express,express_for_target_data,express_for_relabeled_shadow_data,express_for_re_target_data,total_index_num,dir_name,sub_dir_name,category_len,MIA_load_save):
  """
  metric_names,list,the names of metric values
  MIA_count,int,count the number of MIAs
  infer_result,dict,inference result of different index numbers
  t_per_result,dict,performance of attack model on attack dataset
  express,DataFrame, shadow dataset on shadow model   ex_0
  express_for_target_data, DataFrame, target dataset on target model ex_1
  express_for_relabeled_shadow_data, DataFrame, relabeled shadow dataset on relabeled shadow model ex_2
  express_for_re_target_data, DataFrame, target dataset on relabeled shadow model  ex_3
  total_index_num,int,the number of data samples in total dataset
  dir_name,str,dir name of saving the picture of different split
  sub_dir_name,str,dir name of saving the picture for each split

  transfer_sign,int,whether use target model relabel the shadow dataset 1 relabel, 0 not relabel
  separate_threshold,int,whether use different thresholds for different classes 1 use, 0 not use
  class_num,int,the class num under the condition of separate_threshold
  category_len,int,the number of categories
  """
  #gap MIA which is the standard of this type of MIAs
  #different metric vaule
  #express,express_for_target_data / transfer-label-only express_for_relabeled_shadow_data,express_for_re_target_data
  #whether use shadow model to fine-tune threshold or use other way get metric threshold(random data samples' output)
  #one class with one threshold or all data samples with one threshold
  
  MIA_count,infer_result,t_per_result=gap_MIA(express_for_target_data,MIA_count,infer_result,t_per_result,total_index_num)
  
  current_model_path=MIA_load_save['save_model_path'] #end with current time stamp
  split_index=MIA_load_save['split_index']
  pre_MIA_time_stamp=MIA_load_save['pre_MIA_time_stamp']
  save_model=MIA_load_save['save_target_shadow_MIAs']

  previous_thresholds={}
  current_thresholds={}
  if len(pre_MIA_time_stamp)>0:
    #load previous MIAs
    print(f"current_model_path:{current_model_path}")
    pre_model_path=current_model_path[:-17]+pre_MIA_time_stamp+'/'+'split-'+str(split_index)+'/'+'MIA_thresholds.json'
    print(f"pre_model_path:{pre_model_path}")
    if os.path.exists(pre_model_path):
      with open(pre_model_path,'r') as f:
        previous_thresholds=json.load(f)


  for metric_name in metric_names:
    transfer_sign=0
    separate_threshold=0
    class_num=-1
    MIA_count,infer_result,t_per_result,pre_thre=fine_tune_with_shadow_model(metric_name,express,express_for_target_data,transfer_sign,separate_threshold,class_num,MIA_count,infer_result,t_per_result,total_index_num,dir_name,sub_dir_name,previous_thresholds)
    
    for key in pre_thre.keys():
      current_thresholds[key]=pre_thre[key]
    # a=test_if_same(infer_result,total_index_num,dir_name,sub_dir_name)
    # print(a)
    if express_for_relabeled_shadow_data.size != 0 and express_for_re_target_data.size != 0:
      transfer_sign=1
      separate_threshold=0
      class_num=-1
      MIA_count,infer_result,t_per_result,pre_thre=fine_tune_with_shadow_model(metric_name,express_for_relabeled_shadow_data,express_for_re_target_data,transfer_sign,separate_threshold,class_num,MIA_count,infer_result,t_per_result,total_index_num,dir_name,sub_dir_name,previous_thresholds)

      for key in pre_thre.keys():
        current_thresholds[key]=pre_thre[key]

  if save_model and len(list(previous_thresholds.keys()))==0:
    #save current MIA
    current_model_save_path=current_model_path+'/'+'split-'+str(split_index)+'/'
    if not os.path.exists(current_model_save_path):
      os.makedirs(current_model_save_path)
    with open(current_model_save_path+'MIA_thresholds.json','w') as f:
      json.dump(current_thresholds,f)

  return MIA_count,infer_result,t_per_result
    

def gap_MIA(express_for_target_data,MIA_count,infer_result,t_per_result,total_index_num): #judge member just by whether predicted correctly
  #deal with p_v_correctness_based independently  p_v_correctness_based 0 or 1
  index_list=[item for item in express_for_target_data.index]
  predict_=express_for_target_data['p_v_correctness_based'].tolist()
  true_label=express_for_target_data['label'].tolist()
  
  key_str="gap_MIA"
  MIA_count=MIA_count+1
  
  #inference situation of different indexs
  if key_str not in infer_result.keys():
    infer_result[key_str]=[]
    for i in range(0,total_index_num):
      infer_result[key_str].append([[],[],[],[]])
  for i in range(0,len(index_list)):
    index=index_list[i]
    if true_label[i]==1:
      infer_result[key_str][index][3].append(-1)
      if predict_[i]==true_label[i]:
        infer_result[key_str][index][0].append(1)
        infer_result[key_str][index][2].append(1)
      else:
        infer_result[key_str][index][0].append(0)
        infer_result[key_str][index][2].append(0)
    elif true_label[i]==0:
      infer_result[key_str][index][2].append(-1)
      if predict_[i]==true_label[i]:
        infer_result[key_str][index][1].append(1)
        infer_result[key_str][index][3].append(1)
      else:
        infer_result[key_str][index][1].append(0)
        infer_result[key_str][index][3].append(0)
  for index_num in range(0,total_index_num):
    if index_num not in index_list:
      infer_result[key_str][index_num][2].append(-1)
      infer_result[key_str][index_num][3].append(-1)

  e_acc=accuracy_score(true_label,predict_)
  e_pre=precision_score(true_label,predict_,zero_division=0)
  e_rec=recall_score(true_label,predict_,zero_division=0)

  #attack performance
  c_m=confusion_matrix(true_label,predict_)
  TP=c_m[1][1]
  TN=c_m[0][0]
  FP=c_m[0][1]
  FN=c_m[1][0]
  t_per_result[key_str]=[TP,TN,FP,FN,e_acc,e_pre,e_rec]
  
  print("**gap_MIA**eval TP:%d TN:%d FP:%d FN:%d accuracy:%.3f precision:%.3f recall:%.3f"%(TP,TN,FP,FN,e_acc,e_pre,e_rec))
  return MIA_count,infer_result,t_per_result



#define the function of applying MIA with one of metric value, fine-tune the threshold value on shadow data and evaluate on target data
def fine_tune_with_shadow_model(metric_name,express,express_for_target_data,transfer_sign,separate_threshold,class_num,MIA_count,infer_result,t_per_result,total_index_num,dir_name,sub_dir_name,previous_thresholds):
  """
  metric_name,str,the name of metric value
  express,DataFrame, shadow dataset on shadow model   ex_0
  express_for_target_data, DataFrame, target dataset on target model ex_1
  transfer_sign,int,whether use target model relabel the shadow dataset 1 relabel, 0 not relabel
  separate_threshold,int,whether use different thresholds for different classes 1 use, 0 not use
  class_num,int,the class num under the condition of separate_threshold

  MIA_count,int,count the number of MIAs
  infer_result,dict,inference result of different index numbers
  t_per_result,dict,performance of attack model on attack dataset
  total_index_num,int,the number of data samples in total dataset
  dir_name,str,dir name of saving the picture of different split
  sub_dir_name,str,dir name of saving the picture for each split

  (not use)express_for_relabeled_shadow_data, DataFrame, relabeled shadow dataset on relabeled shadow model ex_2
  (not use)express_for_re_target_data, DataFrame, target dataset on relabeled shadow model  ex_3
  (not use)index_lists, list, different index lists under MIA eg. [[index_list1],[index_list2],[index_list3],[index_list4]] m_m(member be predicted as member),m_nom, nom_nom, nom_m
  """

  temp_1=express[['label',metric_name]]#shadow data
  temp_2=express_for_target_data[['label',metric_name]] #target data
  if temp_1.shape[0]>0 and temp_2.shape[0]>0:
    MIA_count,infer_result,t_per_result,pre_thre=threshold_and_evaluation(temp_1,temp_2,metric_name,transfer_sign,separate_threshold,MIA_count,infer_result,t_per_result,total_index_num,previous_thresholds)
  del temp_1,temp_2
  del express,express_for_target_data

  return MIA_count,infer_result,t_per_result,pre_thre


def map_to_list(map1):
  x=[]
  y=[]
  for item in map1.keys():
    x.append(item)
    y.append(map1[item])
  return x,y

#define the function to show difference
def show_difference(m_data_shadow,nm_data_shadow,m_data_target,nm_data_target,metric_name,transfer_sign,separate_threshold,class_num,dir_name,sub_dir_name,x_label,y_label,title):
  x_1,y_1=map_to_list(m_data_shadow)
  x_2,y_2=map_to_list(nm_data_shadow)
  x_3,y_3=map_to_list(m_data_target)
  x_4,y_4=map_to_list(nm_data_target)
  if separate_threshold==1:
    pic_name=metric_name+"-"+str(transfer_sign)+"-"+str(separate_threshold)+"-"+str(class_num)+".png"
  else:
    pic_name=metric_name+"-"+str(transfer_sign)+"-"+str(separate_threshold)+".png"
  dir_t_name="./result/"+dir_name+"/"+sub_dir_name+"/difference_pic/"
  if not os.path.exists(dir_t_name):
    os.makedirs(dir_t_name)
  
  fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)
  axs[0].plot(x_1,y_1,'bo',label = "member in shadow model") #,'m_data_shadow'  ,'nm_data_shadow'
  axs[0].plot(x_1,y_1,'bo',label = "non-member in shadow model") #,'m_data_shadow'  ,'nm_data_shadow'

  axs[1].plot(x_3,y_3,'g.',label = "member in target model")
  axs[1].plot(x_4,y_4,'k*',label = "non-member in target model") #,'m_data_target'  ,'nm_data_target'
  axs[0].set_xlabel(x_label)
  axs[0].set_ylabel(y_label)
  axs[1].set_xlabel(x_label)
  axs[1].set_ylabel(y_label)
  plt.title(title)
  plt.savefig(os.path.join(dir_t_name,pic_name), dpi=150)
  plt.clf()

#define the function of finding threshold with shadow data and applying the threshold into the target data
def threshold_and_evaluation(temp_1,temp_2,metric_name,transfer_sign,separate_threshold,MIA_count,infer_result,t_per_result,total_index_num,previous_thresholds):
  """
  temp_1,pd.DataFrame, shadow data true_label(1 member,0 non-member)
  temp_2,pd.DataFrame,{[true_label, metric_value],.....} for target data
  transfer_sign,int,whether use target model relabel the shadow dataset 1 relabel, 0 not relabel
  separate_threshold,int,whether use different thresholds for different classes 1 use, 0 not use
  MIA_count,int,count the number of MIAs
  infer_result,dict,inference result of different index numbers
  t_per_result,dict,performance of attack model on attack dataset
  total_index_num,int,the number of data samples in total dataset
  """
  
  classifier_or_not=0 #1 classifier 0 non-classifier
  key_str="MIA-"+str(classifier_or_not)+"-"+metric_name+"-"+str(transfer_sign)+"-"+str(separate_threshold)

  if key_str in previous_thresholds:
    pre_=previous_thresholds[key_str]
    max_acc_1, max_acc_threshold=pre_[0],pre_[1]
  else:
    shadow_x=temp_1[metric_name].tolist() #metric value
    shadow_y=temp_1['label'].tolist() #true label
    print("length of shadow_x:%d shadow_y:%d"%(len(shadow_x),len(shadow_y)))
    del temp_1
    fpr, tpr, thresholds = roc_curve(shadow_y, shadow_x, pos_label=1)
    acc_1=[]
    for thresh in thresholds:
      if metric_name in ['entropy','normalized_entropy','loss','CELoss_in_z','mentr']:
        t_=[1 if m < thresh else 0 for m in shadow_x]
        acc_1.append(accuracy_score(shadow_y,t_))
      elif metric_name in ['ground_truth_p','max_p_value','logit_gap','conf_gap']:
        t_=[1 if m > thresh else 0 for m in shadow_x]
        acc_1.append(accuracy_score(shadow_y,t_))
    max_acc_1=np.array(acc_1).max()
    max_acc_threshold=thresholds[np.array(acc_1).argmax()]
  
  pre_thre={}
  pre_thre[key_str]=[max_acc_1,max_acc_threshold]

  target_x=temp_2[metric_name].tolist()#metric value
  target_y=temp_2['label'].tolist()#true label
  print("length of target_x:%d target_y:%d"%(len(target_x),len(target_y)))
  

  t_2_indexs=[item for item in temp_2.index]
  del temp_2
  #evaluate target data with the threshold found with shadow data
  if metric_name in ['entropy','normalized_entropy','loss','CELoss_in_z','mentr']:
    t_=[1 if m < max_acc_threshold else 0 for m in target_x]
    e_acc=accuracy_score(target_y,t_)
    e_pre=precision_score(target_y,t_,zero_division=0)
    e_rec=recall_score(target_y,t_,zero_division=0)
      
  elif metric_name in ['ground_truth_p','max_p_value','logit_gap','conf_gap']:
    t_=[1 if m > max_acc_threshold else 0 for m in target_x]
    e_acc=accuracy_score(target_y,t_)
    e_pre=precision_score(target_y,t_,zero_division=0)
    e_rec=recall_score(target_y,t_,zero_division=0)
  
  
  MIA_count=MIA_count+1
  #inference situation of different indexs
  if key_str not in infer_result.keys():
    infer_result[key_str]=[]
    for i in range(0,total_index_num):
      infer_result[key_str].append([[],[],[],[]])
      
  for i in range(0,len(t_2_indexs)):
    index=t_2_indexs[i]
    if target_y[i]==1:
      infer_result[key_str][index][3].append(-1)
      if t_[i]==target_y[i]:
        infer_result[key_str][index][0].append(1)
        infer_result[key_str][index][2].append(1)
      else:
        infer_result[key_str][index][0].append(0)
        infer_result[key_str][index][2].append(0)
    elif target_y[i]==0:
      infer_result[key_str][index][2].append(-1)
      if t_[i]==target_y[i]:
        infer_result[key_str][index][1].append(1)
        infer_result[key_str][index][3].append(1)
      else:
        infer_result[key_str][index][1].append(0)
        infer_result[key_str][index][3].append(0)
  for index_num in range(0,total_index_num):
    if index_num not in t_2_indexs:
      infer_result[key_str][index_num][2].append(-1)
      infer_result[key_str][index_num][3].append(-1)

  #attack performance
  c_m=confusion_matrix(target_y,t_)
  TP=c_m[1][1]
  TN=c_m[0][0]
  FP=c_m[0][1]
  FN=c_m[1][0]
  t_per_result[key_str]=[TP,TN,FP,FN,e_acc,e_pre,e_rec]
  
  print("Thresholds found on shadow data: Acc=%.3f"%(max_acc_threshold))
  print("Max evalu-metric in shadow data: Acc=%.3f"%(max_acc_1))
  print("**"+metric_name+"**eval transfer-label-only:%d mulitple-threshold:%d TP:%d TN:%d FP:%d FN:%d accuracy:%.3f precision:%.3f recall:%.3f"%(transfer_sign,separate_threshold,TP,TN,FP,FN,e_acc,e_pre,e_rec))
  return MIA_count,infer_result,t_per_result,pre_thre
    