import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score
from xgboost import XGBClassifier
import numpy as np
import pickle
import os


from utils import spilt_attack_train_and_test
from model import model_attack
from train_process import train_one_attack_dataset

#define the function for evaluating attack model on the attack features gained from target dataset
def eval_compute(feature_sign,model_name,MIA_count,true_label,model_prediction,index_list,infer_result,t_per_result,total_index_num):
  #for example, #i "SVM" MIA_count true_label svm_predict index_list infer_result t_per_result
  """
  feature_sign,int,subscript of array total_x_list to represent features used for training attack model
  model_name,str,the type of attack_model
  MIA_count,int,count the number of MIAs
  true_label,list,true lable of attack dataset
  model_prediction,list,prediction label of attack model on attack dataset
  index_list,list,index number of data selected for evaluation
  infer_result,dict,inference result of different index numbers
  t_per_result,dict,performance of attack model on attack dataset
  total_index_num,int,the number of data samples in total dataset
  """
  classifier_or_not=1 #1 classifier 0 non-classifier
  key_str="MIA-"+str(classifier_or_not)+"-"+model_name+"-"+str(feature_sign)+"-"+str(MIA_count)
  MIA_count=MIA_count+1
  if key_str not in infer_result.keys():
    infer_result[key_str]=[]
    for i in range(0,total_index_num):
      infer_result[key_str].append([[],[],[],[]])
  si=0
  for i in range(0,len(index_list)):
    index=index_list[i]
    if true_label[i]==1:
      infer_result[key_str][index][3].append(-1) #not in non-member
      if model_prediction[i]==true_label[i]:
        infer_result[key_str][index][0].append(1)
        infer_result[key_str][index][2].append(1)
      else:
        infer_result[key_str][index][0].append(0)
        infer_result[key_str][index][2].append(0)
    elif true_label[i]==0:
      infer_result[key_str][index][2].append(-1) #not in member
      if model_prediction[i]==true_label[i]:
        infer_result[key_str][index][1].append(1)
        infer_result[key_str][index][3].append(1)
      else:
        infer_result[key_str][index][1].append(0)
        infer_result[key_str][index][3].append(0)
  for index_num in range(0,total_index_num):
    if index_num not in index_list: #not in member and non-memer, in the shadow/target dataset
      infer_result[key_str][index_num][2].append(-1) #not in member
      infer_result[key_str][index_num][3].append(-1) #not in non-member
  #compute TP TN FP FN accuracy precision recall
  c_m=confusion_matrix(true_label,model_prediction)
  TP=c_m[1][1]
  TN=c_m[0][0]
  FP=c_m[0][1]
  FN=c_m[1][0]
  accuracy=accuracy_score(true_label,model_prediction)
  precision=precision_score(true_label,model_prediction,zero_division=0)
  recall=recall_score(true_label,model_prediction,zero_division=0)
  
  t_per_result[key_str]=[TP,TN,FP,FN,accuracy,precision,recall]

  print("**"+model_name+"**eval TP:%d TN:%d FP:%d FN:%d accuracy:%.3f precision:%.3f recall:%.3f"%(TP,TN,FP,FN,accuracy,precision,recall))
  return MIA_count,infer_result,t_per_result

def load_classic_model(model, model_name,feature_sign,MIA_count,x_train,y_train,load_from_pre, split_index, save_model, pre_model_path, current_model_path):

  classifier_or_not=1 #1 classifier 0 non-classifier
  key_str="MIA-"+str(classifier_or_not)+"-"+model_name+"-"+str(feature_sign)+"-"+str(MIA_count)
  print(f"pre_attack_model_path:{pre_model_path}")
  print(f"current_attack_model_path:{current_model_path}")
  
  if load_from_pre:
    pre_model_file_name=pre_model_path+"/"+key_str+".pkl"
    print(f"pre_attack_model_file_name:{pre_model_file_name}")
    if os.path.exists(pre_model_file_name):
      model=pickle.load(open(pre_model_file_name,"rb"))
      print("load pre attack model")
      return model

  model.fit(x_train,y_train)

  if save_model:
    current_model_file_name=current_model_path+'/'+'split-'+str(split_index)+"/"+key_str+".pkl"
    print(f"current_attack_model_file_name:{current_model_file_name}")
    if not os.path.exists(current_model_path):
      os.makedirs(current_model_path)
    pickle.dump(model,open(current_model_file_name,"wb"))
    print("save current attack model")

  return model




def distinguish_with_classifier(MIA_count,infer_result,t_per_result,total_x_list,y_list,express,express_for_target_data,device,total_index_num,MIA_load_save):
  """
  MIA_count,int,signal of current MIA
  t_per_result,dict,the performance of MIA like "MIA_1":[TP,TN,FP,FN,accuracy,precision,recall]#one dict in performance_result
  infer_result,dict,the inference result of data samples like "MIA_1":[[[correctly_infer_time_while_as_member,,,],[correctly_infer_time_while_as_non-member,,,]],,,]#length=total_index_num
  total_x_list,list, features list
  y_list,list,label
  express,pd.DataFrame,shadow dataset's attack features on shadow model   ex_0
  express_for_target_data,pd.DataFrame,target dataset's attack features on target model ex_1
  (not use)express_for_relabeled_shadow_data,pd.DataFrame,relabeled shadow dataset's attack features on relabeled shadow model ex_2
  (not use)express_for_re_target_data,pd.DataFrame,target dataset's attack features on relabeled shadow model  ex_3
  """
  # load_from_pre, save_model, pre_model_path, current_model_path

  current_model_path=MIA_load_save['save_model_path'] #end with current time stamp
  split_index=MIA_load_save['split_index']
  pre_MIA_time_stamp=MIA_load_save['pre_MIA_time_stamp']
  save_model=MIA_load_save['save_target_shadow_MIAs']

  pre_model_path=""
  load_from_pre=False
  if len(pre_MIA_time_stamp)>0:
    #load previous MIAs
    print(f"Current--current_model_path:{current_model_path}")
    pre_model_path=current_model_path[:-17]+pre_MIA_time_stamp+'/'+'split-'+str(split_index)
    load_from_pre=True
  

  for i in range(0,len(total_x_list)):
    x_list=total_x_list[i]
    print("The features used for classification: %s"%(str(x_list)))
    train_dataset,test_dataset,X_train,X_test,y_train,y_test = spilt_attack_train_and_test(express,test_size=0.30,random_seed=i,x_list=x_list,y_list='label')

    true_label=express_for_target_data['label'].tolist()
    index_list=[item for item in express_for_target_data.index]

    #use classical models firstly (SVM)
    model=SVC(C=0.8,kernel='rbf',class_weight='balanced')
    model=load_classic_model(model,'SVM',i,MIA_count,X_train,y_train,load_from_pre, split_index, save_model, pre_model_path, current_model_path)
    test_accuracy=accuracy_score(y_test, model.predict(X_test))
    print("**SVM**test accuracy:%.3f****"%(test_accuracy))
    svm_predict=[int(item) for item in model.predict(express_for_target_data.loc[:,x_list]).tolist()]
    print("SVM prediction: length:%d pre[0]:%s"%(len(svm_predict),str(svm_predict[0])))
    MIA_count,infer_result,t_per_result=eval_compute(i,"SVM",MIA_count,true_label,svm_predict,index_list,infer_result,t_per_result,total_index_num)
    
    #use linear classification
    model=SGDClassifier(loss='log_loss')
    model=load_classic_model(model,'LC',i,MIA_count,X_train,y_train,load_from_pre, split_index, save_model, pre_model_path, current_model_path)
    test_accuracy=accuracy_score(y_test, model.predict(X_test))
    print("**LC**test accuracy:%.3f****"%(test_accuracy))
    lc_predict=[int(item) for item in model.predict(express_for_target_data.loc[:,x_list]).tolist()]
    print("LC prediction: length:%d pre[0]:%s"%(len(lc_predict),str(lc_predict[0])))
    MIA_count,infer_result,t_per_result=eval_compute(i,"LC",MIA_count,true_label,lc_predict,index_list,infer_result,t_per_result,total_index_num)

    #use XGBoost
    model=XGBClassifier(max_depth=10,random_state=42,verbosity=0)
    model=load_classic_model(model,'XGBoost',i,MIA_count,X_train,y_train,load_from_pre, split_index, save_model, pre_model_path, current_model_path)
    test_accuracy=accuracy_score(y_test, model.predict(X_test))
    print("**XGBoost**test accuracy:%.3f****"%(test_accuracy))
    xgb_predict=[int(item) for item in model.predict(express_for_target_data.loc[:,x_list]).tolist()]
    print("XGBoost prediction: length:%d pre[0]:%s"%(len(xgb_predict),str(xgb_predict[0])))
    MIA_count,infer_result,t_per_result=eval_compute(i,"XGBoost",MIA_count,true_label,xgb_predict,index_list,infer_result,t_per_result,total_index_num)

    #use Neural Network
    print("*********Neural Network********")
    attack_lr=0.01
    attack_batch_size=36
    attack_epoch_num=50
    attack_model=model_attack(input_n=len(x_list),hidden_n=len(x_list),output_n=1).to(device)
    attack_criterion=torch.nn.BCELoss()
    attack_optimizer=optim.SGD(attack_model.parameters(), lr=attack_lr, momentum=0.9)
    attack_metric='binary_accuracy'

    load_=False
    classifier_or_not=1 #1 classifier 0 non-classifier
    key_str="MIA-"+str(classifier_or_not)+"-"+"NN"+"-"+str(i)+"-"+str(MIA_count)
    if load_from_pre:
      pre_model_file_name=pre_model_path+"/"+key_str+".pth"
      print(f"pre_NN_based_attack_model_file_name:{pre_model_file_name}")
      if os.path.exists(pre_model_file_name):
        attack_model.load_state_dict(torch.load(pre_model_file_name,map_location=device))
        load_=True
        print("Load pre_NN_based_attack_model!")
    
    if not load_:
      attack_model=train_one_attack_dataset(train_dataset,test_dataset,attack_epoch_num,attack_batch_size,attack_model,x_list,y_list,device,attack_optimizer,attack_criterion,attack_metric)
    
    if save_model and not load_:
      torch.save(attack_model.state_dict(), current_model_path+'/'+'split-'+str(split_index)+"/"+key_str+".pth")
      print("Save NN_based_attack_model!")
    #eval data performance
    r_d_size=express_for_target_data.shape[0]
    t_3=express_for_target_data[x_list].values
    eval_f=torch.from_numpy(t_3).float().to(device)
    del t_3
    eval_output=attack_model(eval_f)
    del eval_f
    #change eval_f 
    t_predict=eval_output.cpu().detach().numpy().tolist()
    nn_predict=[1 if item[0] >0.5 else 0 for item in t_predict]
    print("NN prediction: length:%d pre[0]:%s"%(len(nn_predict),str(nn_predict[0])))
    MIA_count,infer_result,t_per_result=eval_compute(i,"NN",MIA_count,true_label,nn_predict,index_list,infer_result,t_per_result,total_index_num)

  return MIA_count,infer_result,t_per_result
