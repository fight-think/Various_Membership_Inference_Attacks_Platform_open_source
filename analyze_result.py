import matplotlib
from numpy.core import multiarray
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import figure
import os
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool,Manager
from datetime import datetime
from copy import deepcopy
from cycler import cycler



from utils import draw_a_pic

def test_shape_of_infer_result(infer_result,total_index_num):
  key_list=[item for item in infer_result.keys()]
  key_len=len(key_list)
  print("%d MIAs: %s"%(key_len,str(key_list)))
  

  infer_list=[]
  for i in range(0,total_index_num):
    infer_list.append([[],[]]) #adjust the order of inference result

  print("Adjust axis and combine list.....")
  si=0
  for key in tqdm(infer_result.keys()):
    for index_num in range(0,total_index_num):
      infer_list[index_num][0].append(infer_result[key][index_num][0])
      infer_list[index_num][1].append(infer_result[key][index_num][1])
  del infer_list
  del infer_result
  


def show_current_time():
  now = datetime.now() # current date and time
  date_time = now.strftime("%Y%d%m%H%M%S")
  print("Current time:%s"%(date_time))

def test_if_same(infer_result,total_index_num,dir_name,sub_dir_name):
  key_list=[item for item in infer_result.keys()]
  key_len=len(key_list)
  print("%d MIAs: %s"%(key_len,str(key_list)))
  
  infer_list=[[[],[]]]*total_index_num #adjust the order of inference result

  print("Adjust axis and combine list.....")
  for key in tqdm(infer_result.keys()):
    for index_num in range(0,total_index_num):
      infer_list[index_num][0].append(infer_result[key][index_num][0])
      infer_list[index_num][1].append(infer_result[key][index_num][1])
  
  print("Analyze list.....")
  for index_num in tqdm(range(0,total_index_num)):
    if len(infer_list[index_num][0][0])>0:
      #check the items in infer_list[index_num][0]
      len_list=[len(item) for item in infer_list[index_num][0]]
      t_list=[len_list[0]]*len(len_list)

      if len_list!=t_list:
        print("Not all elements in list same")
        t_set=set(len_list)-set(t_list)
        print("different:%s"%(str(t_set)))

        for diff in t_set:
          idx=len_list.index(diff)
          print("another one:")
          if idx>0:
            print(str(infer_list[index_num][0][idx-1]))
          else:
            print(str(infer_list[index_num][0][idx+1]))
          print("current one:")
          print(str(infer_list[index_num][0][idx]))

        print("indifference in training data")
        return False
    
    if len(infer_list[index_num][0][1])>0:
      #check the items in infer_list[index_num][1]
      len_list=[len(item) for item in infer_list[index_num][1]]
      t_list=[len_list[0]]*len(len_list)

      if len_list!=t_list:
        print("Not all elements in list same")
        t_set=set(len_list)-set(t_list)
        print("different:%s"%(str(t_set)))

        for diff in t_set:
          idx=len_list.index(diff)
          print("another one:")
          if idx>0:
            print(str(infer_list[index_num][1][idx-1]))
          else:
            print(str(infer_list[index_num][1][idx+1]))
          print("current one:")
          print(str(infer_list[index_num][1][idx]))

        print("indifference in testing data")
        return False
  return True

#define the function run in each process of the Pool
def compute_fun_2(index_num,index_data,dis_Index_as_x,train_MIA,test_MIA,train_empty_list,test_empty_list):
  if len(index_data[0][0])>0:
    t_0=np.array(index_data[0])
    temp=np.count_nonzero(t_0,axis=0)/len(t_0)
    train_avg=np.average(temp)
    train_max=np.max(temp)
    train_min=np.min(temp)
    #for MIA
    temp=np.count_nonzero(t_0,axis=1)/len(t_0[0])
    del t_0
    train_MIA.put((index_num,temp.tolist()))
  else:
    train_empty_list.put(index_num)

  if len(index_data[1][0])>0:
    t_1=np.array(index_data[1])
    temp=np.count_nonzero(t_1,axis=0)/len(t_1)
    test_avg=np.average(temp)
    test_max=np.max(temp)
    test_min=np.min(temp)
    #for MIA
    temp=np.count_nonzero(t_1,axis=1)/len(t_1[0])
    del t_1
    test_MIA.put((index_num,temp.tolist()))
  else:
    test_empty_list.put(index_num)
    
  del temp
  
  dis_Index_as_x.put((index_num,[train_avg,train_max,train_min,test_avg,test_max,test_min]))
  print("One index number is done......")
  show_current_time()


#analyze the infer_result
"""
{
  "MIA_1":[[[correctly_infer_time_while_as_member,,,],[correctly_infer_time_while_as_non-member,,,]],,,]#length=total_index_num,
  "MIA_2":.....#initialize as [[[],[]]]*total_index_num if "MIA_i" not in the key list
}
"""
def analyze_infer_result(infer_result,total_index_num,dir_name,sub_dir_name):

  key_list=[item for item in infer_result.keys()]
  key_len=len(key_list)
  print("%d MIAs: %s"%(key_len,str(key_list)))
  
  infer_list=[]
  for i in range(0,total_index_num):
    infer_list.append([[],[]])
  #infer_list=[[[],[]]]*total_index_num #adjust the order of inference result

  #2024/9/21 filter MIAs with relatively high inference accuracy for analyzing the vulnerable data points

  print("Adjust axis and combine list.....")
  si=0
  for key in tqdm(infer_result.keys()):
    for index_num in range(0,total_index_num):
      infer_list[index_num][0].append(infer_result[key][index_num][0])
      infer_list[index_num][1].append(infer_result[key][index_num][1])
      # if index_num==0:
      #   print(np.shape(np.array(infer_list[index_num][0])))
  del infer_result
  #x=index_num y=inference_rate(average inference_rate)
  #[hightest training inference_rate, lowest training inference_rate, average training vulnerability, hightest testing inference_rate, lowest testing inference_rate, average testing vulnerability]
  dis_Index_as_x=[]
  for i in range(0,total_index_num):
    dis_Index_as_x.append([])
  # mana=Manager()
  # dis_Index_as_x_q=mana.Queue()
  dis_MIA_as_x=[]
  for i in range(0,key_len):
    dis_MIA_as_x.append([])
  train_MIA=[-1]*total_index_num
  #train_MIA_q=mana.Queue()
  # train_MIA_total=[]
  test_MIA=[-1]*total_index_num
  #test_MIA_q=mana.Queue()
  # test_MIA_total=[]

  train_empty_list=[]
  #train_empty_list_q=mana.Queue()
  test_empty_list=[]
  #test_empty_list_q=mana.Queue()

  sign=0

  print("Analyze list.....")
  # infer_list=[[[[MIA_1],[MIA_2],...],[[MIA_1],[MIA_2],...]],index_1, index_2, ....]
  # [MIA_i] might be empty or [1,0,1,0....]; len([MIA_1])=len([MIA_2])=...
  for index_num in tqdm(range(0,total_index_num)):
    if len(infer_list[index_num][0][0])>0:
      t_0=np.array(infer_list[index_num][0])
      temp=np.count_nonzero(t_0,axis=0)/len(t_0)
      train_avg=np.average(temp)
      train_max=np.max(temp)
      train_min=np.min(temp)
      #for MIA
      temp=np.count_nonzero(t_0,axis=1)/len(t_0[0])
      del t_0
      train_MIA[index_num]=temp.tolist()
      del temp
    else:
      train_avg=-1
      train_max=-1
      train_min=-1
      train_empty_list.append(index_num)

    if len(infer_list[index_num][1][0])>0:
      t_1=np.array(infer_list[index_num][1])
      temp=np.count_nonzero(t_1,axis=0)/len(t_1)
      test_avg=np.average(temp)
      test_max=np.max(temp)
      test_min=np.min(temp)
      #for MIA
      temp=np.count_nonzero(t_1,axis=1)/len(t_1[0])
      del t_1
      test_MIA[index_num]=temp.tolist()
      del temp
    else:
      test_avg=-1
      test_max=-1
      test_min=-1
      test_empty_list.append(index_num)

    dis_Index_as_x[index_num]=[index_num,train_avg,train_max,train_min,test_avg,test_max,test_min]
  dis_Index_as_x=[item for item in dis_Index_as_x if -1 not in item]
  
  #select 0.01 data samples as vulnerable ones
  select_num=int(total_index_num*0.01)
  if select_num > len(dis_Index_as_x):
    select_num=len(dis_Index_as_x)
  def fun_1(x):
    return x[4] #sort as test_avg
  def fun_2(x):
    return x[1] #sort as train_avg
  dis_Index_as_x.sort(key=fun_1,reverse=True)
  vul_as_test_avg=[dis_Index_as_x[i][0] for i in range(0,select_num)]
  dis_Index_as_x.sort(key=fun_2,reverse=True)
  vul_as_train_avg=[dis_Index_as_x[i][0] for i in range(0,select_num)] #draw as train_avg decrease
  #t_train_MIA=deepcopy(train_MIA)
  #t_test_MIA=deepcopy(test_MIA)

  #train_MIA=[[MIA_1_acc, MIA_2_acc, ...],-1,[MIA_1_acc, MIA_2_acc, ...],...]
  train_MIA=[item for item in train_MIA if item != -1]
  test_MIA=[item for item in test_MIA if item != -1]

  train_MIA=np.array(train_MIA)
  test_MIA=np.array(test_MIA)

  train_avg_MIA=np.average(train_MIA,axis=0)
  train_max_MIA=np.max(train_MIA,axis=0)
  train_min_MIA=np.min(train_MIA,axis=0)

  test_avg_MIA=np.average(test_MIA,axis=0)
  test_max_MIA=np.max(test_MIA,axis=0)
  test_min_MIA=np.min(test_MIA,axis=0)

  for i in range(0,key_len):
    dis_MIA_as_x[i]=[train_avg_MIA[i],train_max_MIA[i],train_min_MIA[i],test_avg_MIA[i],test_max_MIA[i],test_min_MIA[i]]

  
  print("The index numbers not used in training data:")
  print(str(train_empty_list))
  print(len(train_empty_list))
  print("The index numbers not used in testing data:")
  print(str(test_empty_list))
  print(len(test_empty_list))
  
  #in the comparison, only display the one with all metrics computed
  final_dis_index_as_x=[[item[1],item[2],item[3],item[4],item[5],item[6]] for item in dis_Index_as_x if -1 not in item]
  final_dis_MIA_as_x=[item for item in dis_MIA_as_x if -1 not in item]

  #display
  line_labels=["train_avg(vulnerability)","train_max","train_min","test_avg(vulnerability)","test_max","test_min"]
  x_tick_num=10
  x_ticks=[]
  pic_name="inference rate comparison from sample"
  x_label="mark of sample"
  y_label="inference rate"
  pic_type='2_1'
  draw_a_pic(line_labels,final_dis_index_as_x,x_tick_num,x_ticks,pic_name,dir_name,sub_dir_name,x_label,y_label,pic_type)

  line_labels=["train_avg_MIA","train_max_MIA","train_min_MIA","test_avg_MIA","test_max_MIA","test_min_MIA"]
  x_tick_num=10 #not use
  x_ticks=[]
  pic_name="inference rate comparison from MIA"
  x_label="MIA type"
  y_label="inference rate"
  pic_type='2_1'
  draw_a_pic(line_labels,dis_MIA_as_x,x_tick_num,x_ticks,pic_name,dir_name,sub_dir_name,x_label,y_label,pic_type)

  return vul_as_test_avg,vul_as_train_avg

  #hightest inference rate under MIAs of total data samples as training and testing dataset

  #the training vulnerability of total data samples

  #the test vulnerability of total data samples

  #comparasion of different MIAs, MIA type as the X axis, the inference rates as the y values(training and testing)
  
  #make the model not well-generalized, and compute the change of vulnerability of different MIAs(change datasize, split rates)

  #find the data samples with high vulnerability and compare them with outher detection and Long's work(try to find the regularity)

#analyze the performance_result
"""
[{
  "target_model":[select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss]
  "shadow_model":[select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss]
  "shadow_model_with_relabelled_data":[select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss]
  "MIA_1":[TP,TN,FP,FN,accuracy,precision,recall],
  "MIA_2":.....
},,,,,]
"""
def analyze_performance_result(performance_result,dir_name,sub_dir_name):
  #analyze train_acc of target_model, shadow_model, and shadow_model_with_relabelled_data

  model_acc=[]
  for i in range(0,len(performance_result)):
    t_1=performance_result[i]['target_model'][1]
    t_2=performance_result[i]['target_model'][3]
    t_3=performance_result[i]['shadow_model'][1]
    t_4=performance_result[i]['shadow_model'][3]
    if 'shadow_model_with_relabelled_data' in performance_result[i].keys():
      t_5=performance_result[i]['shadow_model_with_relabelled_data'][1]
      t_6=performance_result[i]['shadow_model_with_relabelled_data'][3]
      model_acc.append([t_1,t_2,t_3,t_4,t_5,t_6])
    else:
      model_acc.append([t_1,t_2,t_3,t_4])
  #draw model_acc
  line_labels=["target_train_acc","target_test_acc","shadow_train_acc","shadow_test_acc","relabelled_shadow_train_acc","relabelled_shadow_train_acc"]
  x_tick_num=10 #not use
  x_ticks=[i for i in range(0,len(model_acc))]
  pic_name="evaluation rate comparison of non-attack models"
  x_label="different split"
  y_label="evaluation rate"
  pic_type='3_1'
  #not draw due to the possible miss of t_5 and t_6 (2023.7.10)
  #draw_a_pic(line_labels,model_acc,x_tick_num,x_ticks,pic_name,dir_name,sub_dir_name,x_label,y_label,pic_type)

  MIA_acc=[]
  MIA_pre=[]
  MIA_rec=[]
  key_list=[item for item in performance_result[0].keys() if item not in ['target_model','shadow_model','shadow_model_with_relabelled_data']]
  #train_acc of different MIA, different splitations with different lines
  for key in key_list:
    t=[]
    t_1=[]
    t_2=[]
    for i in range(0,len(performance_result)):
      t.append(performance_result[i][key][4])
      t_1.append(performance_result[i][key][5])
      t_2.append(performance_result[i][key][6])
    MIA_acc.append(t)
    MIA_pre.append(t_1)
    MIA_rec.append(t_2)
  #draw MIA_acc
  line_labels=['split-'+str(i) for i in range(0,len(performance_result))]
  x_tick_num=10 #not use
  x_ticks=[]
  pic_name="accuracy comparison of attack models"
  x_label="MIA type"
  y_label="evaluation rate"
  pic_type='1_1'
  draw_a_pic(line_labels,MIA_acc,x_tick_num,x_ticks,pic_name,dir_name,sub_dir_name,x_label,y_label,pic_type)

  
  line_labels=['split-'+str(i) for i in range(0,len(performance_result))]
  x_tick_num=10 #not use
  x_ticks=[]
  pic_name="precision comparison of attack models"
  x_label="MIA type"
  y_label="evaluation rate"
  pic_type='1_1'
  draw_a_pic(line_labels,MIA_pre,x_tick_num,x_ticks,pic_name,dir_name,sub_dir_name,x_label,y_label,pic_type)

  line_labels=['split-'+str(i) for i in range(0,len(performance_result))]
  x_tick_num=10 #not use
  x_ticks=[]
  pic_name="recall comparison of attack models"
  x_label="MIA type"
  y_label="evaluation rate"
  pic_type='1_1'
  draw_a_pic(line_labels,MIA_rec,x_tick_num,x_ticks,pic_name,dir_name,sub_dir_name,x_label,y_label,pic_type)



def analyze_infer_of_one_exp(infer_result,total_index_num,train_avg_des_i,mem_t_as_train_avg_i,test_avg_des_i,non_mem_t_as_test_avg_i,train_test_des_i,mem_t_as_train_test_i,non_mem_t_as_train_test_i,final_dis_MIA_as_x_i,select_MIA_MIR_MT_i,select_MIA_NMIR_NMT_i,MIA_target_analysis=False):

  key_list=[item for item in infer_result.keys()]
  key_len=len(key_list)
  print("%d MIAs: %s"%(key_len,str(key_list)))
  
  infer_list=[]
  for i in range(0,total_index_num):
    if MIA_target_analysis==True:
      infer_list.append([[],[],[],[]])
    else:
      infer_list.append([[],[]])
  #infer_list=[[[],[]]]*total_index_num #adjust the order of inference result

  print("Adjust axis and combine list.....")
  si=0
  for key in tqdm(infer_result.keys()):
    for index_num in range(0,total_index_num):
      infer_list[index_num][0].append(infer_result[key][index_num][0])
      infer_list[index_num][1].append(infer_result[key][index_num][1])
      if MIA_target_analysis==True:
        infer_list[index_num][2].append(infer_result[key][index_num][2])
        infer_list[index_num][3].append(infer_result[key][index_num][3])
  del infer_result
  #[hightest training inference_rate, lowest training inference_rate, average training vulnerability, hightest testing inference_rate, lowest testing inference_rate, average testing vulnerability]
  dis_Index_as_x=[]
  for i in range(0,total_index_num):
    dis_Index_as_x.append([])
  dis_MIA_as_x=[]
  for i in range(0,key_len):
    dis_MIA_as_x.append([])
  train_MIA=[-1]*total_index_num
  test_MIA=[-1]*total_index_num

  train_empty_list=[]
  test_empty_list=[]

  sign=0

  print("Analyze list.....")
  for index_num in tqdm(range(0,total_index_num)):
    if len(infer_list[index_num][0][0])>0:
      t_0=np.array(infer_list[index_num][0])
      temp=np.count_nonzero(t_0,axis=0)/len(t_0)
      train_avg=np.average(temp)
      train_max=np.max(temp)
      train_min=np.min(temp)
      #for MIA
      temp=np.count_nonzero(t_0,axis=1)/len(t_0[0])
      mem_time=len(t_0[0])
      del t_0
      train_MIA[index_num]=temp.tolist()
      del temp
    else:
      train_avg=-1
      train_max=-1
      train_min=-1
      mem_time=-1
      train_empty_list.append(index_num)

    if len(infer_list[index_num][1][0])>0:
      t_1=np.array(infer_list[index_num][1])
      temp=np.count_nonzero(t_1,axis=0)/len(t_1)
      test_avg=np.average(temp)
      test_max=np.max(temp)
      test_min=np.min(temp)
      #for MIA
      temp=np.count_nonzero(t_1,axis=1)/len(t_1[0])
      non_mem_time=len(t_1[0])
      del t_1
      test_MIA[index_num]=temp.tolist()
      del temp
    else:
      test_avg=-1
      test_max=-1
      test_min=-1
      non_mem_time=-1
      test_empty_list.append(index_num)
    
    train_test=-1
    if train_avg!=-1 and test_avg!= -1:
      train_test=train_avg-test_avg

    dis_Index_as_x[index_num]=[index_num,train_avg,train_max,train_min,test_avg,test_max,test_min,train_test,mem_time,non_mem_time]
  #mt_and_nmt
  mt_and_nmt=[[item[8],item[9]] for item in dis_Index_as_x] #[mem_time,non_mem_time] for each index
  
  dis_Index_as_x=[item for item in dis_Index_as_x if -1 not in item]
  
  #select 0.001 data samples as vulnerable ones
  select_num=int(total_index_num*0.001)
  select_num_for_specific_analysis=int(total_index_num) #actually equal len(dis_Index_as_x)
  if select_num > len(dis_Index_as_x):
    select_num=len(dis_Index_as_x)
  def fun_1(x):
    return x[4],x[9] #sort as test_avg
  def fun_2(x):
    return x[1],x[8] #sort as train_avg
  def fun_3(x):
    return x[7] #sort as train_test
  dis_Index_as_x.sort(key=fun_1,reverse=True)
  test_avg_des=[dis_Index_as_x[i][4] for i in range(0,len(dis_Index_as_x))]
  non_mem_t_as_test_avg=[dis_Index_as_x[i][9] for i in range(0,len(dis_Index_as_x))]
  
  if MIA_target_analysis==True:
    test_avg_des_index_num=[dis_Index_as_x[i][0] for i in range(0,len(dis_Index_as_x))]
    non_mem_des_infer_list=[infer_list[index_num][3] for index_num in test_avg_des_index_num] #[[[MIA_1 non_mem [0,1,-1]],[MIA_2 non_mem],...]]

  dis_Index_as_x.sort(key=fun_2,reverse=True)
  train_avg_des=[dis_Index_as_x[i][1] for i in range(0,len(dis_Index_as_x))] #draw as train_avg decrease
  mem_t_as_train_avg=[dis_Index_as_x[i][8] for i in range(0,len(dis_Index_as_x))]

  if MIA_target_analysis==True:
    train_avg_des_index_num=[dis_Index_as_x[i][0] for i in range(0,len(dis_Index_as_x))]
    mem_des_infer_list=[infer_list[index_num][2] for index_num in train_avg_des_index_num] #[[[MIA_1 mem [0,1,-1]],[MIA_2 mem],...]]
  
  dis_Index_as_x.sort(key=fun_3,reverse=True)
  train_test_des=[dis_Index_as_x[i][7] for i in range(0,len(dis_Index_as_x))] #draw as train_test decrease
  mem_t_as_train_test=[dis_Index_as_x[i][8] for i in range(0,len(dis_Index_as_x))]
  non_mem_t_as_train_test=[dis_Index_as_x[i][9] for i in range(0,len(dis_Index_as_x))]

  #t_train_MIA=deepcopy(train_MIA)
  #t_test_MIA=deepcopy(test_MIA)
  train_mem=[[i,mt_and_nmt[i][0]] for i in range(0,len(train_MIA)) if train_MIA[i]!=-1]
  test_nmem=[[i,mt_and_nmt[i][1]] for i in range(0,len(test_MIA)) if test_MIA[i]!=-1]
  train_MIA=[item for item in train_MIA if item != -1]
  test_MIA=[item for item in test_MIA if item != -1]

  train_MIA=np.array(train_MIA)
  test_MIA=np.array(test_MIA)

  train_avg_MIA=np.average(train_MIA,axis=0)
  train_max_MIA=np.max(train_MIA,axis=0)
  train_min_MIA=np.min(train_MIA,axis=0)
  #determine the MIA with median AMIR
  select_MIA_amir=np.percentile(train_avg_MIA,50,interpolation='nearest')
  select_MIA_index_m=(train_avg_MIA.tolist()).index(select_MIA_amir)#find the index of that MIA
  select_MIA_IR_to_points_m=[[train_MIA[i][select_MIA_index_m],train_mem[i][0],train_mem[i][1]] for i in range(0,np.shape(train_MIA)[0])]
  show_data_points_m=[item for item in select_MIA_IR_to_points_m if item[0]>=0.8 and item[2]>=5]
  #sort as the IR
  def fun_4(x):
    return x[0],x[2]
  show_data_points_m.sort(key=fun_4,reverse=True)
  if len(show_data_points_m)>10: #just select up to 10 data points
    show_data_points_m=show_data_points_m[:10]
  show_data_points_m=[[item[0],item[2]] for item in show_data_points_m] #remove index, keep MIR and MT
  select_MIA_MIR_MT=[select_MIA_index_m,select_MIA_amir,show_data_points_m]

  test_avg_MIA=np.average(test_MIA,axis=0)
  test_max_MIA=np.max(test_MIA,axis=0)
  test_min_MIA=np.min(test_MIA,axis=0)
  #determine the MIA with median AMIR
  select_MIA_anmir=np.percentile(test_avg_MIA,50,interpolation='nearest')
  select_MIA_index_nm=(test_avg_MIA.tolist()).index(select_MIA_anmir)#find the index of that MIA
  select_MIA_IR_to_points_nm=[[test_MIA[i][select_MIA_index_nm],test_nmem[i][0],test_nmem[i][1]] for i in range(0,np.shape(test_MIA)[0])]
  show_data_points_nm=[item for item in select_MIA_IR_to_points_nm if item[0]>=0.8 and item[2]>=5]

  show_data_points_nm.sort(key=fun_4,reverse=True)
  if len(show_data_points_nm)>10: #just select up to 10 data points
    show_data_points_nm=show_data_points_nm[:10]
  show_data_points_nm=[[item[0],item[2]] for item in show_data_points_nm] #remove index, keep NMR and NMT
  select_MIA_NMIR_NMT=[select_MIA_index_nm,select_MIA_anmir,show_data_points_nm]

  for i in range(0,key_len):
    #dis_MIA_as_x[i]=[train_avg_MIA[i],train_max_MIA[i],train_min_MIA[i],test_avg_MIA[i],test_max_MIA[i],test_min_MIA[i]]
    dis_MIA_as_x[i]=[train_avg_MIA[i],test_avg_MIA[i]]

  
  print("The index numbers not used in training data:")
  print(str(train_empty_list))
  print(len(train_empty_list))
  print("The index numbers not used in testing data:")
  print(str(test_empty_list))
  print(len(test_empty_list))
  
  #in the comparison, only display the one with all metrics computed
  final_dis_MIA_as_x=[item for item in dis_MIA_as_x if -1 not in item]

  train_avg_des_i.append(train_avg_des)
  mem_t_as_train_avg_i.append(mem_t_as_train_avg)#[0:select_num]
  test_avg_des_i.append(test_avg_des)
  non_mem_t_as_test_avg_i.append(non_mem_t_as_test_avg)#[0:select_num]
  train_test_des_i.append(train_test_des)
  mem_t_as_train_test_i.append(mem_t_as_train_test)#[0:select_num]
  non_mem_t_as_train_test_i.append(non_mem_t_as_train_test)#[0:select_num]
  final_dis_MIA_as_x_i.append(final_dis_MIA_as_x)
  select_MIA_MIR_MT_i.append(select_MIA_MIR_MT)
  select_MIA_NMIR_NMT_i.append(select_MIA_NMIR_NMT)


  #analyze the specific target model and MIA with non_mem_des_infer_list and mem_des_infer_list 2023.7.8
  #mem_des_infer_list   #[[[MIA_1 mem [0,1,-1]],[MIA_2 mem],...]] len=the number of selected points 40(selected number)*54(the number of MIA)*20(split_num)
  #non_mem_des_infer_list  #[[[MIA_1 non_mem [0,1,-1]],[MIA_2 non_mem],...]] len=the number of selected points 40(selected number)*54(the number of MIA)*20(split_num)

  if MIA_target_analysis==True:
    print("np.shape(np.array(mem_des_infer_list)):"+str(np.shape(np.array(mem_des_infer_list))))
    mem_des_infer_list=deepcopy(np.transpose(np.array(mem_des_infer_list),(1,2,0))) #change to 54*20*40000
    mem_count_0=np.sum(mem_des_infer_list==0,axis=2) #count the number of data points
    mem_count_1=np.sum(mem_des_infer_list==1,axis=2)
    mem_count_0_1=mem_count_0+mem_count_1
    
    mem_accuracy_on_vulnerable=np.divide(mem_count_1,mem_count_0_1,out=np.ones_like(mem_count_0_1)*-1.0,where=mem_count_0_1!=0) #54*20
    mem_accuracy_on_vulnerable=deepcopy(np.transpose(mem_accuracy_on_vulnerable,(1,0))) #change to 20*54

    mem_count_0_1=deepcopy(np.transpose(mem_count_0_1,(1,0)))
    print("np.shape(mem_count_0_1):"+str(np.shape(mem_count_0_1)))
    print("the number of data points in vulnerable ones occur in the member data of each target model:")
    print(str(mem_count_0_1.tolist()))
    print("member list:")
    print([item[0] for item in mem_count_0_1.tolist()])

    print("mem_accuracy_on_vulnerable:")
    print(mem_accuracy_on_vulnerable.tolist())

    
    print("np.shape(np.array(non_mem_des_infer_list)):"+str(np.shape(np.array(non_mem_des_infer_list))))
    non_mem_des_infer_list=deepcopy(np.transpose(np.array(non_mem_des_infer_list),(1,2,0)))
    non_mem_count_0=np.sum(non_mem_des_infer_list==0,axis=2) #count the number of data points
    non_mem_count_1=np.sum(non_mem_des_infer_list==1,axis=2)
    non_mem_count_0_1=non_mem_count_0+non_mem_count_1
    non_mem_accuracy_on_vulnerable=np.divide(non_mem_count_1,non_mem_count_0_1,out=np.ones_like(non_mem_count_0_1)*-1.0,where=non_mem_count_0_1!=0) #54*20
    non_mem_accuracy_on_vulnerable=deepcopy(np.transpose(non_mem_accuracy_on_vulnerable,(1,0))) #change to 20*54

    non_mem_count_0_1=deepcopy(np.transpose(non_mem_count_0_1,(1,0)))
    print("np.shape(non_mem_count_0_1):"+str(np.shape(non_mem_count_0_1)))
    print("the number of data points in vulnerable ones occur in the non-member data of each target model:")
    print(str(non_mem_count_0_1.tolist()))
    print("non-member list:")
    print([item[0] for item in non_mem_count_0_1.tolist()])

    print("non_mem_accuracy_on_vulnerable:")
    print(non_mem_accuracy_on_vulnerable.tolist())
    
    return train_avg_des_i,mem_t_as_train_avg_i,test_avg_des_i,non_mem_t_as_test_avg_i,train_test_des_i,mem_t_as_train_test_i,non_mem_t_as_train_test_i,final_dis_MIA_as_x_i,select_MIA_MIR_MT_i,select_MIA_NMIR_NMT_i,mem_accuracy_on_vulnerable,non_mem_accuracy_on_vulnerable

  return train_avg_des_i,mem_t_as_train_avg_i,test_avg_des_i,non_mem_t_as_test_avg_i,train_test_des_i,mem_t_as_train_test_i,non_mem_t_as_train_test_i,final_dis_MIA_as_x_i,select_MIA_MIR_MT_i,select_MIA_NMIR_NMT_i

#analyze the performance_result
"""
[{
  "target_model":[select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss]
  "shadow_model":[select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss]
  "shadow_model_with_relabelled_data":[select_epoch,select_train_acc,select_train_loss,select_test_acc,select_test_loss]
  "MIA_1":[TP,TN,FP,FN,accuracy,precision,recall],
  "MIA_2":.....
},,,,,]
"""
def analyze_per_of_one_exp(performance_result,MIA_acc_of_datasets,table_list_1):
  #analyze train_acc of target_model, shadow_model, and shadow_model_with_relabelled_data
  table_list=[]
  model_acc=[]
  for i in range(0,len(performance_result)):
    t_1=performance_result[i]['target_model'][1]
    t_2=performance_result[i]['target_model'][3]
    t_3=performance_result[i]['shadow_model'][1]
    t_4=performance_result[i]['shadow_model'][3]
    if 'shadow_model_with_relabelled_data' in performance_result[i].keys(): #no relabelled model while training with different datasets
      t_5=performance_result[i]['shadow_model_with_relabelled_data'][1]
      t_6=performance_result[i]['shadow_model_with_relabelled_data'][3]
      model_acc.append([t_1,t_2,t_3,t_4,t_5,t_6])
    else:
      model_acc.append([t_1,t_2,t_3,t_4])
  target_train=[item[0] for item in model_acc]
  target_test=[item[1] for item in model_acc]
  train_test=[item[0]-item[1] for item in model_acc]
  print("the gap between the train and test data in target models:")
  print(str(train_test))
  train_acc_avg=np.mean(np.array(target_train))
  table_list.append(train_acc_avg)
  train_acc_var=np.var(np.array(target_train))
  table_list.append(train_acc_var)
  test_acc_avg=np.mean(np.array(target_test))
  table_list.append(test_acc_avg)
  test_acc_var=np.var(np.array(target_test))
  table_list.append(test_acc_var)
  train_test_avg=np.mean(np.array(train_test))
  table_list.append(train_test_avg)
  train_test_var=np.var(np.array(train_test))
  table_list.append(train_test_var)

  MIA_acc=[]
  key_list=[item for item in performance_result[0].keys() if item not in ['target_model','shadow_model','shadow_model_with_relabelled_data']]
  #train_acc of different MIA, different splitations with different lines
  for key in key_list:
    t=[]

    for i in range(0,len(performance_result)):
      t.append(performance_result[i][key][4])
    
    MIA_acc.append(np.mean(np.array(t)))
  
  MIA_acc_avg=np.mean(np.array(MIA_acc))
  MIA_acc_var=np.var(np.array(MIA_acc))
  MIA_acc_med=np.median(np.array(MIA_acc))
  MIA_acc_max=max(MIA_acc)
  MIA_acc_min=min(MIA_acc)
  
  table_list.append(MIA_acc_avg)
  table_list.append(MIA_acc_var)
  table_list.append(MIA_acc_med)
  table_list.append(MIA_acc_max)
  table_list.append(MIA_acc_min)

  MIA_acc_of_datasets.append(MIA_acc)
  table_list_1.append(table_list)
  return MIA_acc_of_datasets,table_list_1

if __name__ == "__main__":
  
  train_avg_des,mem_t_as_train_avg,test_avg_des,non_mem_t_as_test_avg,train_test_des,mem_t_as_train_test,non_mem_t_as_train_test,final_dis_MIA_as_x=[],[],[],[],[],[],[],[]
  
  MIA_acc_of_datasets=[]
  table_list=[] #[[train_acc_avg,train_acc_var,test_acc_avg,test_acc_var,train_test_avg,train_test_var,MIA_acc_avg,MIA_acc_var,MIA_acc_med,MIA_acc_max,MIA_acc_min]]
  select_MIA_MIR_MT=[]
  select_MIA_NMIR_NMT=[]

  total_index_num=40000
  common_path='/XXXXX/membership_inference_attack/Various_Membership_Inference_Attacks_Platform/result/'
  subdir_path='default_exploration_cifar'
  cifar_1_infer_path=common_path+f"{subdir_path}/CIFAR-10-CIFAR-10-model_cifar_LetNet-model_cifar_LetNet-40000-40000-20-1-0.5-0.5-0.5-2024-05-09-195709/split-19/infer_result_8586.txt"
  with open(cifar_1_infer_path,'rb')as f:
    infer_result=pickle.load(f)
  train_avg_des,mem_t_as_train_avg,test_avg_des,non_mem_t_as_test_avg,train_test_des,mem_t_as_train_test,non_mem_t_as_train_test,final_dis_MIA_as_x,select_MIA_MIR_MT,select_MIA_NMIR_NMT=analyze_infer_of_one_exp(infer_result,total_index_num,train_avg_des,mem_t_as_train_avg,test_avg_des,non_mem_t_as_test_avg,train_test_des,mem_t_as_train_test,non_mem_t_as_train_test,final_dis_MIA_as_x,select_MIA_MIR_MT,select_MIA_NMIR_NMT)

  cifar_1_per_path=common_path+f"{subdir_path}/CIFAR-10-CIFAR-10-model_cifar_LetNet-model_cifar_LetNet-40000-40000-20-1-0.5-0.5-0.5-2024-05-09-195709/split-19/per_result_170568.txt"
  with open(cifar_1_per_path,'rb')as f:
    performance_result=pickle.load(f)
  MIA_acc_of_datasets,table_list=analyze_per_of_one_exp(performance_result,MIA_acc_of_datasets,table_list)
  