import numpy as np
import pandas as pd
import math
import torch
from copy import deepcopy
from torch.func import functional_call, vmap, grad


#define the function for obtaining mterics of model on dataset
def compute_metric(y,row,ex,e_criterion,min_p_float,device,model_name):
  """
  y,tensor, prediction output
  row, pd.Serice, the input of the model
  ex,dic,save the metric and return
  e_criterion, torch.nn.CrossEntropyLoss(), the loss function
  min_p_float, float, the minest positive float number in the environment
  device, torch.device, cup or gpu
  """
  basic_col_names=[]
  metric_col_names=[]
  class_num=y.shape[1]
  t=torch.argmax(y,dim=1)
  ground_truth=row['label']
  
  #view the class with hightest probability as the true label sine the adversa
  loss = e_criterion(y,torch.from_numpy(row[['label']].values).long().to(device))
  
  normalized_entroy=0.0
  mentr_value=0.0
  max_p_value=-1
  
  if model_name=='model_mnist':
    length_=10
  elif model_name=='model_cifar':
    length_=10
  elif model_name=='model_cifar_100':
    length_=100
  elif model_name=='model_purchase':
    length_=100

  for i in range(0,length_):
    p_v=y[0][i].item()
    if p_v>max_p_value:
      max_p_value=p_v
    ex[str(i)]=p_v
    basic_col_names.append(str(i))
    if p_v>0:
      normalized_entroy+=p_v*(math.log(p_v))
    #compute Mentr value
    if i==ground_truth:
      if p_v<min_p_float:
        mentr_value+=(-(1-p_v)*(math.log(min_p_float)))
      else:
        mentr_value+=(-(1-p_v)*(math.log(p_v)))
    else:
      if (1-p_v)<min_p_float:
        mentr_value+=(-p_v)*(math.log(min_p_float))
      else:
        mentr_value+=(-p_v)*(math.log(1-p_v))
      
  cross_entroy=-normalized_entroy
  normalized_entroy=(-1)*normalized_entroy/(math.log(class_num))
  #compute the other statistical metrics about the prediction of model
  
  #prediction correctness (which is only tested on target model)
  if t[0]==row['label']:
    ex['p_v_correctness_based']=1
  else:
    ex['p_v_correctness_based']=0
  metric_col_names.append('p_v_correctness_based')
  #cross entropy of the model’s prediction vector
  ex['cross_entroy']=cross_entroy
  metric_col_names.append('cross_entroy')
  #normalized entropy of the model’s prediction vector
  ex['normalized_entroy']=normalized_entroy
  metric_col_names.append('normalized_entroy')
  ex['ground_truth_p']=y[0][int(ground_truth)].item()
  metric_col_names.append('ground_truth_p')
  ex['mentr']=mentr_value
  metric_col_names.append('mentr')
  
  ex['loss']=loss.item()
  metric_col_names.append('loss')
  if ex['ground_truth_p']>min_p_float:
    CELoss_in_z=-(math.log(ex['ground_truth_p']))
  else:
    CELoss_in_z=-(math.log(min_p_float))
  ex['CELoss_in_z']=CELoss_in_z
  metric_col_names.append('CELoss_in_z')
  ex['max_p_value']=max_p_value
  metric_col_names.append('max_p_value')
  ex['class_num']=int(ground_truth)
  
  return ex,basic_col_names,metric_col_names


def obtain_feature_and_metric(express,train_data_t,test_data_t,model,feature_c_name,e_criterion,min_p_float,device,model_name,category_len):
  """
  express,pd.DataFrame(),store the features and metircs
  train_data_t,pd.DataFrame(), training data of model [feature,label_i,label,original_label(relabelled_shadow_data)]
  test_data_t,pd.DataFrame(),test data of model
  model, nn.model, the model for computing features and metrics
  feature_c_name, list of str, the column names of features
  e_criterion, torch.nn.CrossEntropyLoss(), the loss function
  min_p_float, float, the minest positive float number in the environment
  device, torch.device, cpu or gpu
  model_name, string, the name of the model
  category_len,int,the number of categories in this dataset
  """
  train_data=deepcopy(train_data_t)
  test_data=deepcopy(test_data_t)

  basic_col_names=[]
  metric_col_names=[]
  train_data['_label']=train_data['label'] #store previous label
  test_data['_label']=test_data['label']
  train_data['class_num']=train_data['label']
  test_data['class_num']=test_data['label']

  if 'model_mnist' in model_name:
    temp_train=train_data[feature_c_name].values.reshape(train_data.shape[0],1,28,28)
    temp_test=test_data[feature_c_name].values.reshape(test_data.shape[0],1,28,28)
    # temp=np.expand_dims(temp,axis=1)
  elif 'model_cifar' in model_name:
    temp_train=train_data[feature_c_name].values.reshape(train_data.shape[0],3,32,32)
    temp_test=test_data[feature_c_name].values.reshape(test_data.shape[0],3,32,32)
  elif 'model_cifar_100' in model_name:
    temp_train=train_data[feature_c_name].values.reshape(train_data.shape[0],3,32,32)
    temp_test=test_data[feature_c_name].values.reshape(test_data.shape[0],3,32,32)
  elif 'model_purchase' in model_name:
    temp_train=train_data[feature_c_name].values.reshape(train_data.shape[0],600)
    temp_test=test_data[feature_c_name].values.reshape(test_data.shape[0],600)
  feature_train=torch.from_numpy(temp_train).float().to(device)
  feature_test=torch.from_numpy(temp_test).float().to(device)
  
  avg_loss=torch.nn.CrossEntropyLoss()
  # 2024/8/20 define function to obtain gradients
  def compute_loss(params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)
    model.eval()
    _, predictions = functional_call(model, (params, buffers), (batch,))
    loss = avg_loss(predictions, targets)
    return loss
  model.eval()
  print(f"After model.eval()--list(model.parameters())[0].requires_grad:{list(model.parameters())[0].requires_grad}")

  train_gradients=[]
  test_gradients=[]
  print(f"Enable gredient--list(model.parameters())[0].requires_grad:{list(model.parameters())[0].requires_grad}")
  params = {k: v.detach() for k, v in model.named_parameters()}
  buffers = {k: v.detach() for k, v in model.named_buffers()}
  ft_compute_grad = grad(compute_loss)
  ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0),randomness='same')
  
  total_feature_train=feature_train
  total_target_train=torch.from_numpy(train_data['_label'].values).to(device)
  total_feature_test=feature_test
  total_target_test=torch.from_numpy(test_data['_label'].values).to(device)
  keep_gradient_num=100
    
  def obtain_gradients(total_feature,total_target,gradients):
    batch_size=100 #batch size for extracting gradients
    start_index=0
    while start_index<total_feature.shape[0]:
      end_index=min(start_index+batch_size,total_feature.shape[0])
      t_feature=total_feature[start_index:end_index]
      t_target=total_target[start_index:end_index]
      ft_per_sample_grads = ft_compute_sample_grad(params, buffers, t_feature, t_target)

      combine_top_100=[]
      for key in ft_per_sample_grads.keys():
        if 'weight' in key:
          
          shape0=ft_per_sample_grads[key].shape[0]
          shape1=ft_per_sample_grads[key].reshape((shape0,-1)).shape[1]
          k_=min(keep_gradient_num,shape1)
          top_100=torch.topk(ft_per_sample_grads[key].reshape((shape0,-1)),k=k_,dim=1)[0]
          combine_top_100.append(top_100)
      combine_top_100=torch.cat(combine_top_100,dim=1)
      shape1=combine_top_100.shape[1]
      top_100_gradients=torch.topk(combine_top_100,k=min(keep_gradient_num,shape1),dim=1)[0]
      gradients.append(top_100_gradients.cpu())
      
      start_index=end_index
    return gradients
  train_gradients=obtain_gradients(total_feature_train,total_target_train,train_gradients)
  test_gradients=obtain_gradients(total_feature_test,total_target_test,test_gradients)
  
  train_gradients=np.concatenate(train_gradients,axis=0)
  print(f"np.shape(train_gradients):{np.shape(train_gradients)}")
  test_gradients=np.concatenate(test_gradients,axis=0)
  print(f"np.shape(test_gradients):{np.shape(test_gradients)}")
  assert train_gradients.shape[0]==train_data.shape[0] and test_gradients.shape[0]==test_data.shape[0]
  assert train_gradients.shape[1]==test_gradients.shape[1]==keep_gradient_num
  gradient_labels=[]
  for i in range(0,train_gradients.shape[1]):
    gradient_labels.append(f'grad_{i}')
  basic_col_names+=gradient_labels

  train_grad = pd.DataFrame(data = train_gradients, index = [item for item in train_data.index], columns = gradient_labels)
  test_grad= pd.DataFrame(data = test_gradients, index = [item for item in test_data.index], columns = gradient_labels)
  del train_gradients,test_gradients

  with torch.no_grad():
    print(f"No gradient--list(model.parameters())[0].requires_grad:{list(model.parameters())[0].requires_grad}")
    y_train,y_train_logits=model(feature_train)
    y_test,y_test_logits=model(feature_test)

  logits_labels=[]
  for i in range(0,y_train_logits.shape[1]):
    logits_labels.append(f'logit_{i}')
  basic_col_names+=logits_labels

  train_logits = pd.DataFrame(data = y_train_logits.cpu().numpy(), index = [item for item in train_data.index], columns = logits_labels)
  test_logits= pd.DataFrame(data = y_test_logits.cpu().numpy(), index = [item for item in test_data.index], columns = logits_labels)
  
  
  train_data['predict_label']=torch.argmax(y_train,dim=1).cpu().numpy()
  test_data['predict_label']=torch.argmax(y_test,dim=1).cpu().numpy()
  train_data['max_p_value']=torch.max(y_train,dim=1,keepdim=True).values.cpu().numpy()
  test_data['max_p_value']=torch.max(y_test,dim=1,keepdim=True).values.cpu().numpy()
  metric_col_names.append('max_p_value')

  if type(e_criterion)==torch.nn.modules.loss.CrossEntropyLoss:
    train_data['loss']=e_criterion(y_train_logits.cpu(), torch.from_numpy(train_data['_label'].values)).numpy()
    test_data['loss']=e_criterion(y_test_logits.cpu(), torch.from_numpy(test_data['_label'].values)).numpy()
    metric_col_names.append('loss')

  t_1=y_train.cpu().tolist()
  t_2=train_data['_label'].tolist()
  train_data['ground_truth_p']=[t_1[i][t_2[i]] for i in range(0,len(t_2))]
  t_1=y_test.cpu().tolist()
  t_2=test_data['_label'].tolist()
  test_data['ground_truth_p']=[t_1[i][t_2[i]] for i in range(0,len(t_2))]
  metric_col_names.append('ground_truth_p')

  prob_labels=[]
  for i in range(0,category_len):
    prob_labels.append(str(i))
  basic_col_names+=prob_labels

  train_prob = pd.DataFrame(data = y_train.cpu().numpy(), index = [item for item in train_data.index], columns = prob_labels)
  test_prob= pd.DataFrame(data = y_test.cpu().numpy(), index = [item for item in test_data.index], columns = prob_labels)

  train_prob['entropy']=np.sum(np.where(train_prob[prob_labels]>min_p_float,-train_prob[prob_labels]*np.log(train_prob[prob_labels]),-train_prob[prob_labels]*np.log(min_p_float)),axis=1)
  train_prob['normalized_entropy']=train_prob['entropy']/(np.log(category_len))
  test_prob['entropy']=np.sum(np.where(test_prob[prob_labels]>min_p_float,-test_prob[prob_labels]*np.log(test_prob[prob_labels]),-test_prob[prob_labels]*np.log(min_p_float)),axis=1)
  test_prob['normalized_entropy']=test_prob['entropy']/(np.log(category_len))
  metric_col_names.append('entropy')
  metric_col_names.append('normalized_entropy')
  
  train_data=pd.concat((train_data,train_prob,train_grad,train_logits),axis=1)
  test_data=pd.concat((test_data,test_prob,test_grad,test_logits),axis=1)
  train_data['mentr']=0.0
  test_data['mentr']=0.0
  for i in range(0,category_len):
    
    t=train_data.loc[train_data['_label']==i]
    train_data.loc[train_data['_label']==i,'mentr']=t['mentr']+np.where(t[str(i)]>min_p_float,-(1-t[str(i)])*np.log(t[str(i)]),-(1-t[str(i)])*np.log(min_p_float))
    t=train_data.loc[train_data['_label']!=i]
    train_data.loc[train_data['_label']!=i,'mentr']=t['mentr']+np.where(t[str(i)]>min_p_float,-(t[str(i)])*(1-np.log(t[str(i)])),-(t[str(i)])*(1-np.log(min_p_float)))
    
    t=test_data.loc[test_data['_label']==i]
    test_data.loc[test_data['_label']==i,'mentr']=t['mentr']+np.where(t[str(i)]>min_p_float,-(1-t[str(i)])*np.log(t[str(i)]),-(1-t[str(i)])*np.log(min_p_float))
    t=test_data.loc[test_data['_label']!=i]
    test_data.loc[test_data['_label']!=i,'mentr']=t['mentr']+np.where(t[str(i)]>min_p_float,-(t[str(i)])*(1-np.log(t[str(i)])),-(t[str(i)])*(1-np.log(min_p_float)))
  metric_col_names.append('mentr')
  

  #2024/8/19 add other attack features to increase the variety of MIAs
  # logit-scaled confidence
  # hinge loss
  def obtain_logit_scaled_confidence_and_hinge_loss(logits, conf_scores, labels):
    conf_of_g_t=conf_scores.gather(1,labels.unsqueeze(1)).reshape(-1)
    logit_of_g_t=logits.gather(1,labels.unsqueeze(1)).reshape(-1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[range(len(logits)), labels]=False
    logits_max_left=torch.max(torch.masked_select(logits, mask).reshape(len(logits),-1),dim=1)[0]
    conf_max_left=torch.sum(torch.masked_select(conf_scores, mask).reshape(len(conf_scores),-1),dim=1)
    eps=1e-8
    any_smaller_value_conf_of_g_t=torch.any(torch.lt(conf_of_g_t,eps))
    print(f"any_smaller_value_conf_of_g_t:{any_smaller_value_conf_of_g_t}")
    any_smaller_value_conf_max_left=torch.any(torch.lt(conf_max_left,eps))
    print(f"any_smaller_value_conf_max_left:{any_smaller_value_conf_max_left}")
    conf_of_g_t=torch.clamp(conf_of_g_t, min=eps, max=None)
    conf_max_left=torch.clamp(conf_max_left, min=eps, max=None)
    logit_gap=(logit_of_g_t-logits_max_left).tolist()
    conf_gap=(torch.log(conf_of_g_t)-torch.log(conf_max_left)).tolist()

    return logit_gap, conf_gap
    
  train_data['logit_gap'],train_data['conf_gap']=obtain_logit_scaled_confidence_and_hinge_loss(y_train_logits, y_train, torch.from_numpy(train_data['_label'].values).to(device))
  test_data['logit_gap'],test_data['conf_gap']=obtain_logit_scaled_confidence_and_hinge_loss(y_test_logits, y_test, torch.from_numpy(test_data['_label'].values).to(device))
  metric_col_names.append('logit_gap')
  metric_col_names.append('conf_gap')

  train_data['p_v_correctness_based']=np.where(train_data['predict_label']==train_data['_label'],1,0)
  test_data['p_v_correctness_based']=np.where(test_data['predict_label']==test_data['_label'],1,0)
  
  train_data['label']=1
  test_data['label']=0

  express=pd.concat([train_data,test_data],axis=0)
  #label 1 member,0 non-member; p_v_correctness_based predict_label==label 1,else 0; class_num label in task not attack
  all_label_in_basic=[]
  print(f"len(basic_col_names):{len(basic_col_names)}")
  print(f"basic_col_names:{basic_col_names}")
  for item in basic_col_names:
    all_label_in_basic.append(item)
  keep_labels=['label','p_v_correctness_based','class_num']+metric_col_names+all_label_in_basic


  return express[keep_labels],basic_col_names,metric_col_names  



