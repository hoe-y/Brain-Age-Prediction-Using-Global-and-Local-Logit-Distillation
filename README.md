# Brain-Age-Prediction-Using-Global-and-Local-Logit-Distillation
## 2022-01H Software Convergence Capstone Design
  * 지도 교수 : 이원희 교수님
------------------
### Description
Brain age prediction using global and local logit distillation(GLD).

### 0. Requirements
>apex  
>transformations  
>pytorch  
>numpy  
>logging  
>nibabel  
>sklearn

### 1. Datasets
We use the HCP dataset to train the model.

### 2. Train Models

Train Techer model:
'''shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                        --nproc_per_node=4 \
                        --master_port 51321 \
                        run_gld.py \
                        -b 64 \
                        --epochs 300 \
                        --lr 0.0003 \
                        -p 50  \
                        --arch resnet50 \
                        --data <directory_of_dataset> \
                        --save_root <path_to_save> 
'''

Train Student model by adding Teacher
'''shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                        --nproc_per_node=4 \
                        --master_port 51321 \
                        run_gld.py \
                        -b 64 \
                        --epochs 300 \
                        --lr 0.0003 \
                        -p 50  \
                        --arch resnet18 \
                        --data <directory_of_dataset> \
                        --t_arch resnet50 \
                        --teacher_path <path_of_pretrained_teacher_model>
                        --save_root <path_to_save> 
'''
## Reference
* baseline code : (https://github.com/podismine/BrainAgeReg)
