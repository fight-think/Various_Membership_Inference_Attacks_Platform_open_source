# Various Membership Inference Attacks Platform (VMIAP)

Membership Inference Attack (MIA) has recently become a popular privacy attack in the machine learning field. This repository implements the VMIAP to attack 20 (40) target models trained with different splits of the same dataset with 77 MIAs. Then, the VMIAP will analyze the vulnerability of each data point under those target models and MIAs. We measure the vulnerability of each data point mainly with two metrics: the first one is Average Member Exposure Rate (AMER) and the second one is Average Non-Member Exposure Rate (ANMER). The AMER is the average of rates, each of which is the percentage of MIAs correctly predicting that the target data point is in the training dataset of one target model. If the data point is in the test data, we name it ANMER. For more details, please refer to our [paper](https://arxiv.org/abs/2210.16258).

## Description

### Project Sturcture

    .
    ├── dataset                             # Make sure the existence of datasets before running
    │   ├── mnist-in-csv                    # https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
    │   │   ├── mnist_test.csv              # mnist with CVS format
    │   │   └── mnist_train.csv             # 
    │   ├── purchase-100-in-csv             # https://www.kaggle.com/datasets/datamaters/purchase1000
    │   │   └── purchase100.npz             # purchase100 with npz format
    │   ├── cifar-10-train-in-csv           # https://www.kaggle.com/datasets/fedesoriano/cifar10-python-in-csv             
    │   │   └── train.csv                   # cifar-10 with CSV format
    ├── models                              # The library of models
    ├── privacy_risk_score                  # from https://github.com/inspire-group/membership-inference-evaluation           
    ├── setting                             # each yaml file corrsponding to a setting of run
    ├── shapley_value                       # measure the shapley value of data points
    ├── analyze_result.py                   # analyze the results of multiple MIAs on multiple target models
    ├── attack_feature_and_metric.py        # extract attack features from the model and datasets  
    ├── classifier_based_MIA.py             # the classifier-based MIA
    ├── main.py         
    ├── model.py                            # model definitions for mnist, purchase100, cifar-10
    ├── non_classifier_based_MIA.py         # non-classifier-based MIA
    ├── obtain_dataset.py                   # extract dataset from the storage
    ├── README.md                           
    ├── train_process.py                    # train the model (target, shadow, attack)
    └── utils.py                            # some utility functions

## Getting Started

### Environment

* Python 3.8.17
* Libraries

    > conda install --yes --file requirements.txt
* Default command of running
    > python3 main.py './XXXX/XXXX/config.yaml'(path of a configuration yaml file)

### Configuration

With this repository, you can run experiments under the following settings.

* default_exploration_cifar (default setting)
* default_exploration_mnist
* default_exploration_purchase
* default_hyper_for_target
* more_data_for_training    (more data for training)
* repeat_experiments        (retrain target, shadow, and attack models with different random seeds to initialize the parameters and shuffle the training data)
* shadow_from_other_dis_cifar (data used for shadow models are from different distributions)
* shadow_from_other_dis_mnist
* shadow_from_other_dis_purchase

### Commands of Running

Attack 20 target models (CIFAR-10 and LetNet) with 77 MIAs and analyze the vulnerability of each data point.
> python3 main.py './setting/default_exploration_cifar/LetNet_20_0.5_0.5_0.5.yaml'

### Outputs of Running

Analysis results are saved in the last split (split-19 for 20 target models) and might contain the following files.

* infer_result_XXX.txt (the membership prediction results of 77 MIAs on 20 or 40 target models)
* member_non_member_time.txt (the member and non-member times of each data point)
* per_result_XXX.txt (the performance of target models and MIAs)
* sv_val.txt (the shapley values of data points)
* prs_val.txt (the privacy risk scores of data points)
* vul_as_train_avg.txt (vulnerable data points as AMER)
* vul_as_test_avg.txt (vulnerable data points as ANMER)

Reanalyzing the results of 20 target models with 77 MIAs need to locate infer_result_XXX.txt and per_result_XXX.txt firstly and then run the example shown in analyze_result.py.

### Contact

If you have any questions, please feel free to post an issue or contact me via email (jiaxin.li@studenti.unipd.it).
<!-- 1. (ArcFace + FaceScrub)
> python3 main.py -f_e_shadow_model_name 'arcnet' -f_e_target_model_name 'arcnet'  -f_e_target_epoch_num 150 -f_e_shadow_epoch_num 150 -f_e_shadow_criterion 'additiveangularmarginloss' -f_e_target_criterion 'additiveangularmarginloss' -f_e_shadow_lr 0.0001 -f_e_target_lr 0.0001 -f_e_shadow_input_w 112 -f_e_shadow_input_h 112 -f_e_target_input_w 112 -f_e_target_input_h 112 -f_e_shadow_batch_size 256 -f_e_shadow_num_persons -1 -f_e_shadow_num_images -1 -f_e_target_batch_size 256 -f_e_target_num_persons -1 -f_e_target_num_images -1  -f_e_shadow_one_hot 0 -f_e_target_one_hot 0 -f_e_attack_load_size 1024 -f_e_shadow_output_l 512 -f_e_target_output_l 512 -f_e_attack_one_shadow_with_all_data 1 -f_e_shadow_model_num 1 -f_e_target_users_rate_or_quantity 0.5 -f_e_shadow_users_rate_or_quantity 0.5 -f_e_target_mem_id_rate 0.7 -f_e_target_mem_train_img_rate 0.5 -f_e_shadow_mem_id_rate 0.7 -f_e_shadow_mem_train_img_rate 0.5 -f_e_shadow_dataset_name 'facescrub' -f_e_shadow_img_dir './datasets/facescrub/' -f_e_target_dataset_name 'facescrub' -f_e_target_img_dir './datasets/facescrub/' -f_e_attack_n_jobs 6 -use_pre_split_strategy 0 -use_pre_t_s_models 0 -use_pre_a_features 0 -f_e_train_with_all_data 0 -f_e_attack_target_original 0 -f_e_attack_shadow_original 0 -hyper_search 0 -epochs_per_log 1 -f_e_shadow_train_with_aug 1 -f_e_shadow_test_time_aug 0 -f_e_target_train_with_aug 1 -f_e_target_test_time_aug 0 -f_e_pretrained 0 -f_e_pretrained_test_time_aug 0 -f_e_attack_show_fig 1

2. (DeepFace + FaceScrub)
> python3 main.py -f_e_shadow_model_name 'deepfacenet' -f_e_target_model_name 'deepfacenet'  -f_e_target_epoch_num 100 -f_e_shadow_epoch_num 100 -f_e_shadow_criterion 'crossentropy' -f_e_target_criterion 'crossentropy' -f_e_shadow_input_w 152 -f_e_shadow_input_h 152 -f_e_target_input_w 152 -f_e_target_input_h 152 -f_e_shadow_output_l 4096 -f_e_target_output_l 4096 -f_e_shadow_one_hot 0 -f_e_target_one_hot 0 -f_e_shadow_batch_size 16 -f_e_shadow_num_persons -1 -f_e_shadow_num_images -1 -f_e_target_batch_size 16 -f_e_target_num_persons -1 -f_e_target_num_images -1 -f_e_attack_load_size 32 -f_e_shadow_lr 0.01 -f_e_target_lr 0.01 -f_e_attack_use_cosine 1 -f_e_attack_use_euclidean 0 -f_e_attack_one_shadow_with_all_data 1 -f_e_shadow_model_num 1 -f_e_target_users_rate_or_quantity 0.5 -f_e_shadow_users_rate_or_quantity 0.5 -f_e_target_mem_id_rate 0.8 -f_e_target_mem_train_img_rate 0.5 -f_e_shadow_mem_id_rate 0.8 -f_e_shadow_mem_train_img_rate 0.5 -f_e_shadow_dataset_name 'facescrub' -f_e_shadow_img_dir './datasets/facescrub/' -f_e_target_dataset_name 'facescrub' -f_e_target_img_dir './datasets/facescrub/' -use_pre_split_strategy 0 -use_pre_t_s_models 0 -use_pre_a_features 0 -f_e_attack_n_jobs 6 -f_e_train_with_all_data 0 -hyper_search 0 -epochs_per_log 1 -f_e_shadow_train_with_aug 1 -f_e_shadow_test_time_aug 0 -f_e_target_train_with_aug 1 -f_e_target_test_time_aug 0 -f_e_attack_show_fig 1

3. (SphereFace + FaceScrub)
> python3 main.py -f_e_shadow_model_name 'spherenet' -f_e_target_model_name 'spherenet'  -f_e_target_epoch_num 300 -f_e_shadow_epoch_num 300 -f_e_shadow_criterion 'angleloss' -f_e_target_criterion 'angleloss' -f_e_shadow_input_w 96 -f_e_shadow_input_h 112 -f_e_target_input_w 96 -f_e_target_input_h 112 -f_e_shadow_lr 0.1 -f_e_target_lr 0.1 -f_e_shadow_one_hot 0 -f_e_target_one_hot 0 -f_e_shadow_batch_size 128 -f_e_shadow_num_persons -1 -f_e_shadow_num_images -1 -f_e_target_batch_size 128 -f_e_target_num_persons -1 -f_e_target_num_images -1 -f_e_shadow_output_l 512 -f_e_target_output_l 512 -f_e_attack_load_size 1024 -f_e_attack_use_cosine 1 -f_e_attack_use_euclidean 0 -f_e_attack_one_shadow_with_all_data 1 -f_e_shadow_model_num 1 -f_e_target_users_rate_or_quantity 0.5 -f_e_shadow_users_rate_or_quantity 0.5 -f_e_target_mem_id_rate 0.8 -f_e_target_mem_train_img_rate 0.5 -f_e_shadow_mem_id_rate 0.8 -f_e_shadow_mem_train_img_rate 0.5 -f_e_shadow_dataset_name 'facescrub' -f_e_shadow_img_dir './datasets/facescrub/' -f_e_target_dataset_name 'facescrub' -f_e_target_img_dir './datasets/facescrub/' -use_pre_split_strategy 0 -use_pre_t_s_models 0 -use_pre_a_features 0 -f_e_attack_n_jobs 6 -f_e_train_with_all_data 0 -hyper_search 0 -epochs_per_log 1 -f_e_shadow_train_with_aug 1 -f_e_shadow_test_time_aug 0 -f_e_target_train_with_aug 1 -f_e_target_test_time_aug 0 -f_e_attack_show_fig 1

4. (FaceNet + FaceScrub)
> python3 main.py -f_e_shadow_model_name 'facenet' -f_e_target_model_name 'facenet' -f_e_target_epoch_num 100 -f_e_shadow_epoch_num 100 -f_e_shadow_lr 0.001 -f_e_target_lr 0.001 -f_e_shadow_criterion 'triplet_loss' -f_e_target_criterion 'triplet_loss' -f_e_shadow_input_w 220 -f_e_shadow_input_h 220 -f_e_target_input_w 220 -f_e_target_input_h 220 -f_e_shadow_batch_size 128 -f_e_shadow_num_persons 32 -f_e_shadow_num_images 4 -f_e_target_batch_size 128 -f_e_target_num_persons 32 -f_e_target_num_images 4 -f_e_attack_load_size 1024 -f_e_target_criterion_p1 0.5 -f_e_shadow_criterion_p1 0.5 -f_e_attack_use_squared_euclidean 1 -f_e_attack_use_euclidean 0 -f_e_attack_one_shadow_with_all_data 1 -f_e_shadow_model_num 1 -f_e_target_users_rate_or_quantity 0.5 -f_e_shadow_users_rate_or_quantity 0.5 -f_e_target_mem_id_rate 0.8 -f_e_target_mem_train_img_rate 0.5 -f_e_shadow_mem_id_rate 0.8 -f_e_shadow_mem_train_img_rate 0.5 -f_e_shadow_dataset_name 'facescrub' -f_e_shadow_img_dir './datasets/facescrub/' -f_e_target_dataset_name 'facescrub' -f_e_target_img_dir './datasets/facescrub/' -use_pre_split_strategy 0 -use_pre_t_s_models 0 -use_pre_a_features 0 -f_e_attack_n_jobs 6 -f_e_train_with_all_data 0 -hyper_search 0 -epochs_per_log 1 -f_e_shadow_train_with_aug 1 -f_e_shadow_test_time_aug 0 -f_e_target_train_with_aug 1 -f_e_target_test_time_aug 0 -f_e_attack_show_fig 1

5. (LuNet + FaceScrub)
>python3 main.py -f_e_shadow_model_name 'lunet_rec' -f_e_target_model_name 'lunet_rec'  -f_e_target_epoch_num 100 -f_e_shadow_epoch_num 100 -f_e_shadow_criterion 'triplet_loss_bh' -f_e_target_criterion 'triplet_loss_bh' -f_e_target_lr 0.00001 -f_e_shadow_lr 0.00001 -f_e_shadow_batch_size 128 -f_e_shadow_num_persons 32 -f_e_shadow_num_images 4 -f_e_target_batch_size 128 -f_e_target_num_persons 32 -f_e_target_num_images 4 -f_e_shadow_input_w 64 -f_e_shadow_input_h 128 -f_e_target_input_w 64 -f_e_target_input_h 128 -f_e_target_criterion_p1 0.2 -f_e_shadow_criterion_p1 0.2 -f_e_attack_one_shadow_with_all_data 1 -f_e_shadow_model_num 1 -f_e_target_users_rate_or_quantity 0.5 -f_e_shadow_users_rate_or_quantity 0.5 -f_e_target_mem_id_rate 0.8 -f_e_target_mem_train_img_rate 0.5 -f_e_shadow_mem_id_rate 0.8 -f_e_shadow_mem_train_img_rate 0.5 -f_e_shadow_dataset_name 'facescrub' -f_e_shadow_img_dir './datasets/facescrub/' -f_e_target_dataset_name 'facescrub' -f_e_target_img_dir './datasets/facescrub/' -use_pre_split_strategy 0 -use_pre_t_s_models 0 -use_pre_a_features 0 -f_e_attack_n_jobs 6 -f_e_attack_load_size 1024 -f_e_train_with_all_data 0 -hyper_search 0 -epochs_per_log 1 -f_e_shadow_train_with_aug 1 -f_e_shadow_test_time_aug 0 -f_e_target_train_with_aug 1 -f_e_target_test_time_aug 0 -f_e_attack_show_fig 1

#### Parameters Explanation
1. the model (model_name, loss, the method of loading batch, input, and output size)
    * arcnet
        > -f_e_shadow_model_name 'arcnet' -f_e_target_model_name 'arcnet' -f_e_shadow_criterion 'additiveangularmarginloss' -f_e_target_criterion 'additiveangularmarginloss' -f_e_shadow_num_persons -1 -f_e_shadow_num_images -1 -f_e_target_num_persons -1 -f_e_target_num_images -1 -f_e_shadow_one_hot 0 -f_e_target_one_hot 0 -f_e_shadow_input_w 112 -f_e_shadow_input_h 112 -f_e_target_input_w 112 -f_e_target_input_h 112 -f_e_shadow_output_l 512 -f_e_target_output_l 512
    * deepfacenet
        > -f_e_shadow_model_name 'deepfacenet' -f_e_target_model_name 'deepfacenet' -f_e_shadow_criterion 'crossentropy' -f_e_target_criterion 'crossentropy' -f_e_shadow_num_persons -1 -f_e_shadow_num_images -1 f_e_target_num_persons -1 -f_e_target_num_images -1 -f_e_shadow_one_hot 0 -f_e_target_one_hot 0 -f_e_shadow_input_w 152 -f_e_shadow_input_h 152 -f_e_target_input_w 152 -f_e_target_input_h 152 -f_e_shadow_output_l 4096 -f_e_target_output_l 4096
    * spherenet
        > -f_e_shadow_model_name 'spherenet' -f_e_target_model_name 'spherenet' -f_e_shadow_criterion 'angleloss' -f_e_target_criterion 'angleloss' -f_e_shadow_num_persons -1 -f_e_shadow_num_images -1 -f_e_target_num_persons -1 -f_e_target_num_images -1 -f_e_shadow_one_hot 0 -f_e_target_one_hot 0 -f_e_shadow_input_w 96 -f_e_shadow_input_h 112 -f_e_target_input_w 96 -f_e_target_input_h 112 -f_e_shadow_output_l 512 -f_e_target_output_l 512
    * facenet (f_e_shadow_batch_size=f_e_shadow_num_persons*f_e_shadow_num_images)
        > -f_e_shadow_model_name 'facenet' -f_e_target_model_name 'facenet' -f_e_shadow_criterion 'triplet_loss' -f_e_target_criterion 'triplet_loss' -f_e_shadow_batch_size 128 -f_e_shadow_num_persons 32 -f_e_shadow_num_images 4 -f_e_target_batch_size 128 -f_e_target_num_persons 32 -f_e_target_num_images 4 -f_e_shadow_one_hot -1 -f_e_target_one_hot -1 -f_e_shadow_input_w 220 -f_e_shadow_input_h 220 -f_e_target_input_w 220 -f_e_target_input_h 220 -f_e_shadow_output_l 128 -f_e_target_output_l 128
    * lunet (f_e_shadow_batch_size=f_e_shadow_num_persons*f_e_shadow_num_images)
        > -f_e_shadow_model_name 'lunet_rec' -f_e_target_model_name 'lunet_rec' -f_e_shadow_criterion 'triplet_loss_bh' -f_e_target_criterion 'triplet_loss_bh' -f_e_shadow_batch_size 128 -f_e_shadow_num_persons 32 -f_e_shadow_num_images 4 -f_e_target_batch_size 128 -f_e_target_num_persons 32 -f_e_target_num_images 4 -f_e_shadow_one_hot -1 -f_e_target_one_hot -1 -f_e_shadow_input_w 64 -f_e_shadow_input_h 128 -f_e_target_input_w 64 -f_e_target_input_h 128 -f_e_shadow_output_l 128 -f_e_target_output_l 128

2. the dataset name and folder
    * f_e_shadow_dataset_name 
    * f_e_shadow_img_dir
    * f_e_target_dataset_name 
    * f_e_target_img_dir

3. the dataset split
    * f_e_target_users_rate_or_quantity
    * f_e_shadow_users_rate_or_quantity
    * f_e_target_mem_id_rate
    * f_e_target_mem_train_img_rate
    * f_e_shadow_mem_id_rate
    * f_e_shadow_mem_train_img_rate

4. learning rate and batch size
    * f_e_target_lr
    * f_e_shadow_lr
    * f_e_target_batch_size (for LuNet and FaceNet, batch size is related to the number of persons and images for each person)
    * f_e_shadow_batch_size

5. reuse previous running result
    * use_pre_split_strategy
    * use_pre_t_s_models
    * use_pre_a_features

6. augmentation
    * f_e_shadow_train_with_aug
    * f_e_shadow_test_time_aug
    * f_e_target_train_with_aug
    * f_e_target_test_time_aug

7. the distance between two embedding features
    * f_e_attack_use_squared_euclidean
    * f_e_attack_use_euclidean

8. wandb_api_key
    * wandb_api_key (the api key of your wandb account)
    * wandb_user_name (the user name of your wandb account)

9. testing
    * f_e_train_with_all_data (train the model with all data points in the dataset)
    * hyper_search (seep for searching hyperparameters by setting hyper_serach=1) -->

<!-- ## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46) -->