import torch
import numpy as np
import time
import os

from utils import report_metric,Dataset

#train single target or shadow model
def train_one_dataset(train_data,test_data,epoch_num,batch_size,model,x_list,y_list,device,optimizer,criterion,metric,model_name,dataset_name,model_select_strategy,model_load_save): #return model 
  """
  train_data, DataFrame, the data used for training
  test_data, DataFrame, the data used for testing
  epoch_num, int, the number of epoch of training
  batch_size, int, the number of data samples used for updating weights
  model, torch.nn.Model, the model for training
  x_list, list, the column name of feature
  y_list, list, column name of label
  device, torch.device, the device used for training model
  optimizer, torch.optim.XXX, the optimizer used for training
  criterion, torch.nn.XXX, the loss function
  metric, string, the metric of evaluating
  model_name, string, the name of model, current model_mnist, model_cifar
  model_select_strategy, string, the strategy of picking return model 'highest_train_acc','highest_test_acc','highest_gap','lowest_gap'
  """
  save_model_path=model_load_save['save_model_path'] #end with current time stamp
  split_index=model_load_save['split_index']
  pre_model_time_stamp=model_load_save['pre_model_time_stamp']
  save_current_model=model_load_save['save_target_shadow_MIAs']
  target_shadow_or_reshadow=model_load_save['target_shadow_or_reshadow'] #= 'target', 'shadow', 'reshadow'

  if len(pre_model_time_stamp)>0:
    #load previous model
    print(f"Current--save_model_path:{save_model_path}")
    save_model_path=save_model_path[:-17]+pre_model_time_stamp+'/'+'split-'+str(split_index)+f'/{target_shadow_or_reshadow}_model.pth'
    print(f"Previous--save_model_path:{save_model_path}")
    if os.path.exists(save_model_path):
      print("load model from previous one")
      res_dict=torch.load(save_model_path, map_location=device)
      model.load_state_dict(res_dict['choose_model_state_dict'])
      return model,res_dict['select_epoch'],res_dict['select_train_acc'],res_dict['select_train_loss'],res_dict['select_test_acc'],res_dict['select_test_loss']

  print(f"len(x_list){len(x_list)}")
  print("y_list:%s"%(str(y_list)))
  rows,colus=train_data.shape
  choose_model=model
  if model_select_strategy=='lowest_gap':
    gap=1
  else:
    gap=0
  select_train_acc=0
  select_train_loss=0
  select_test_acc=0
  select_test_loss=0
  select_epoch=0

  choose_sign=0
  start_time=time.time()
  print("Start training......")
  for epoch in range(epoch_num):  # loop over the dataset multiple times
    running_loss = 0.0
    model.train()
    
    train_data=train_data.sample(frac=1) #shuffle data in different epoch
    left_point=0
    while left_point!=rows:
      right_point=min([rows,left_point+batch_size])
      batch_data=train_data.iloc[left_point:right_point,:]

      #the purpose of reshaping is change one demension to 2 demension
      data_size=right_point-left_point
      if 'model_mnist' in model_name:
        temp=batch_data[x_list].values.reshape(data_size,1,28,28)
      elif 'model_cifar' in model_name:
        temp=batch_data[x_list].values.reshape(data_size,3,32,32)
      elif 'model_cifar_100' in model_name:
        temp=batch_data[x_list].values.reshape(data_size,3,32,32)
      elif 'model_purchase' in model_name:
        temp=batch_data[x_list].values.reshape(data_size,600)

      feature=torch.from_numpy(temp).float().to(device)
      del temp

      label=torch.from_numpy(batch_data[y_list].values).float().to(device)

      del batch_data
      left_point=right_point
      
      optimizer.zero_grad()
      # forward + backward + optimize
      outputs,h_f= model(feature) # outputs is confidence scores, h_f is the logits
      del outputs
      if type(criterion)==torch.nn.modules.loss.CrossEntropyLoss:
        loss = criterion(h_f, torch.max(label,1)[1])
      
      loss.backward()
      optimizer.step()
      del h_f
      del label
      
      running_loss += float(loss)
        
    model.eval()
    with torch.set_grad_enabled(False):
      #test data performance
      t_d_size=test_data.shape[0]
      if 'model_mnist' in model_name:
        t_1=test_data[x_list].values.reshape(t_d_size,28,28)
        t_1=np.expand_dims(t_1,axis=1)
      elif 'model_cifar' in model_name:
        t_1=test_data[x_list].values.reshape(t_d_size,3,32,32)
      elif 'model_cifar_100' in model_name:
        t_1=test_data[x_list].values.reshape(t_d_size,3,32,32)
      elif 'model_purchase' in model_name:
        t_1=test_data[x_list].values.reshape(t_d_size,600)
      test_f=torch.from_numpy(t_1).float().to(device)
      del t_1
      test_output,test_h_f= model(test_f)
      test_label=torch.from_numpy(test_data[y_list].values).float().to(device)
      test_acc=report_metric(test_output,test_label,metric)
      test_loss= criterion(test_h_f, torch.max(test_label.float(),1)[1])
      
      #train data performance
      r_d_size=train_data.shape[0]
      if 'model_mnist' in model_name:
        t_2=train_data[x_list].values.reshape(r_d_size,28,28)
        t_2=np.expand_dims(t_2,axis=1)
      elif 'model_cifar' in model_name:
        t_2=train_data[x_list].values.reshape(r_d_size,3,32,32)
      elif 'model_cifar_100' in model_name:
        t_2=train_data[x_list].values.reshape(r_d_size,3,32,32)
      elif 'model_purchase' in model_name:
        t_2=train_data[x_list].values.reshape(r_d_size,600)
      train_f=torch.from_numpy(t_2).float().to(device)
      del t_2
      train_output,train_h_f= model(train_f)
      train_label=torch.from_numpy(train_data[y_list].values).float().to(device)
      train_acc=report_metric(train_output,train_label,metric)
      train_loss = criterion(train_h_f, torch.max(train_label.float(),1)[1])
    
    #select model with different strategies
    #1 highest train_acc
    #2 highest test_acc
    #3 highest train_acc>test_acc meanwhile select_epoch>1/3(epoch_num) if train_acc is always smaller than test_acc return the result of last epoch
    #4 smallest train_acc>test_acc meanwhile select_epoch>1/3(epoch_num) if train_acc is always smaller than test_acc return the result of last epoch
    t_1=model_select_strategy=='highest_gap' and (epoch+1)>=(epoch_num/3) and (train_acc-test_acc)>gap and train_acc>0.8
    t_2=model_select_strategy=='lowest_gap' and (epoch+1)>=(epoch_num/3) and (train_acc-test_acc)>0 and (train_acc-test_acc)<gap and train_acc>0.8
    t_3=model_select_strategy=='highest_train_acc' and train_acc>select_train_acc and train_acc>0.8
    t_4=model_select_strategy=='highest_train_acc' and test_acc>select_test_acc and train_acc>0.8
    
    if (t_1 or t_2 or t_3 or t_4) or (epoch==(epoch_num-1) and choose_sign==0):
      choose_model=model
      select_train_acc=train_acc
      select_train_loss=train_loss.item()
      select_test_acc=test_acc
      select_test_loss=test_loss.item()
      select_epoch=epoch
      gap=train_acc-test_acc
      choose_sign=1
    if epoch==(epoch_num-1):
      print("**************************************************************")
      print('select model infor--epoch: %d, test loss: %.3f, test acc: %.3f, train loss: %.3f, train acc: %.3f' %(select_epoch + 1, select_test_loss, select_test_acc,select_train_loss,select_train_acc))
      print("**************************************************************")
  
  end_time=time.time()
  print("End a training Time spent:%d"%(int(end_time-start_time)))

  if save_current_model:
    save_model_dir=save_model_path+'/'+'split-'+str(split_index)+'/'
    save_model_path=save_model_path+'/'+'split-'+str(split_index)+f'/{target_shadow_or_reshadow}_model.pth'
    print("save model to: "+save_model_path)
    save_dict={
      'choose_model_state_dict':choose_model.state_dict(),
      'select_epoch':select_epoch,
      'select_train_acc':select_train_acc,
      'select_train_loss':select_train_loss,
      'select_test_acc':select_test_acc,
      'select_test_loss':select_test_loss,
    }
    if not os.path.exists(save_model_dir):
      os.makedirs(save_model_dir)
    torch.save(save_dict,save_model_path)

  return choose_model,select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss

#training process of attack model
#define the function of training model
#train single model
def train_one_attack_dataset(train_data,test_data,epoch_num,batch_size,model,x_list,y_list,device,optimizer,criterion,metric): #return model 
  """
  train_data, DataFrame, the data used for training
  test_data, DataFrame, the data used for testing
  #(not use) eval_data, DataFrame, the data used for evaluation eval_data
  epoch_num, int, the number of epoch of training
  batch_size, int, the number of data samples used for updating weights
  model, torch.nn.Model, the model for training
  x_list, list, the column name of feature
  y_list, list, column name of label
  device, torch.device, the device used for training model
  optimizer,torch.optim.XXX, the optimizer used for training
  criterion, torch.nn.XXX, the loss function
  metric, string, the metric of evaluating
  """
  rows,colus=train_data.shape
  best_model=model
  select_train_acc=0
  select_train_loss=0
  select_test_acc=0
  select_test_loss=0
  select_epoch=0

  for epoch in range(epoch_num):  # loop over the dataset multiple times
    running_loss = 0.0
    left_point=0
    train_data=train_data.sample(frac=1) #shuffle data in different epoch
    
    model.train()
    while left_point!=rows:
      right_point=min([rows,left_point+batch_size])
      batch_data=train_data.iloc[left_point:right_point,:]

      #the purpose of reshaping is change one demension to 2 demension
      data_size=right_point-left_point
      temp=batch_data[x_list].values
      feature=torch.from_numpy(temp).float().to(device)
      del temp

      label=torch.from_numpy(batch_data[y_list].values).float().to(device)
      del batch_data
      
      optimizer.zero_grad()
      # forward + backward + optimize
      outputs= model(feature)
      
      #loss = criterion(outputs, torch.max(label,1)[1])
      loss = criterion(outputs, label)
      loss.backward()
      optimizer.step()
      del outputs
      del label
      
      left_point=right_point
      running_loss += float(loss)
        

    model.eval()
    with torch.set_grad_enabled(False):
    #test data performance
      t_d_size=test_data.shape[0]
      t_1=test_data[x_list].values
      test_f=torch.from_numpy(t_1).float().to(device)
      del t_1
      test_output= model(test_f)
      del test_f
      test_label=torch.from_numpy(test_data[y_list].values).float().to(device)
      test_acc=report_metric(test_output,test_label,metric)
      test_loss = criterion(test_output, test_label)
      
      #train data performance
      r_d_size=train_data.shape[0]
      t_2=train_data[x_list].values
      train_f=torch.from_numpy(t_2).float().to(device)
      del t_2
      train_output= model(train_f)
      del train_f
      train_label=torch.from_numpy(train_data[y_list].values).float().to(device)
      train_acc=report_metric(train_output,train_label,metric)
      train_loss = criterion(train_output, train_label)

    if test_acc>select_test_acc:
      best_model=model
      select_train_acc=train_acc
      select_train_loss=train_loss
      select_test_acc=test_acc
      select_test_loss=test_loss
      select_epoch=epoch

    if epoch==(epoch_num-1):
      print("**************************************************************")
      print('select model infor--epoch: %d, test loss: %.3f, test acc: %.3f, train loss: %.3f, train acc: %.3f' %(select_epoch + 1, select_test_loss, select_test_acc,select_train_loss,select_train_acc))
      print("**************************************************************")
  return best_model