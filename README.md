# Brain Age Prediction Using Global and Local Logit Distillation
## 2022-01 Software Convergence Capstone Design

<div align="right">
Advisor : Prof. Won Hee Lee <br>
Department of Software Convergence, Kyung Hee University
</div>

------------------

### Description
**Brain age prediction using global and local logit distillation ([GLD](https://openaccess.thecvf.com/content/ICCV2021/html/Kim_Distilling_Global_and_Local_Logits_With_Densely_Connected_Relations_ICCV_2021_paper.html)).**

  인간의 인지능력은 연령에 비례하여 감소한다. 따라서 MRI 이미지를 통해 노화의 징후를 발견하고 "뇌 나이"를 판단하는 것은 매우 중요하다. 특히, 예측된 뇌 연령과 실제 나이의 차이인 brain age delta는 알츠하이머, 파킨슨 병 등 퇴행성 신경질환을 포함하여 다양한 질병을 예측할 수 있는 중요한 바이오 마커로 활용된다. 최근 딥러닝 기술이 발전하며 기존의 방법보다 훨씬 정확하고 빠르게 뇌 나이를 판단할 수 있게 되었다. 기존의 방법은 정교한 preprocessing (careful bias correction, segmentation 등)을 요구하기 때문에 복잡성으로 인해 임상 환경에서 광범위하게 사용할 수 없었다. 그러나 딥러닝이 등장하면서, 이러한 한계를 뛰어넘는 가능성을 제공하였다.

  그러나 딥러닝의 정확도를 높이기 위해서는 모델을 더 넓고 깊게 만들어야 하므로, 굉장히 많은 연산량을 요구한다는 단점이 존재한다. 이를 해결하기 위해 Knowledge Distillation, Pruning, Quantization 등 다양한 model compression 방법들이 등장하였다. 
  
본 과제에서는 다양한 Knowledge Distillation 방법 중 ICCV 2021의 “Distilling Global and Local Logits with Densely Connected Relations” 논문에서 소개된 Global and Local logit Distillation ([GLD](https://openaccess.thecvf.com/content/ICCV2021/html/Kim_Distilling_Global_and_Local_Logits_With_Densely_Connected_Relations_ICCV_2021_paper.html)) 방법을 **3D ResNet** 모델에 적합하도록 하여 Brain Age Prediction 모델을 compression하는 방법을 제안한다.

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

Train Teacher model:

```shell
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
```

Train Student model by adding Teacher : 

```shell
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
                        --teacher_path <path_of_pretrained_teacher_model> \
                        --save_root <path_to_save> 
```

### 3. Results

Evaluation Metrics : MAE

Setup  | Compression Type  |  Teacher Network  | Student Network  |  # of params teacher  |  # of params student  
:---:|:---:|:---:|:---:|:---:|:---:|
(a)  | Depth  |  ResNet 50  | ResNet 18  |  46.715 M  |  33.327 M
(b)  | Depth  |  ResNet 101  | ResNet 50  |  85.761 M  |  46.715 M


Setup  | Teacher  |  Baseline  | Ours  
:---:|:---:|:---:|:---:
(a)  | 2.85 |  2.89  | 2.68  
(b)  | 2.64  |  2.85  | 2.75  


### 4. Conclusion
결론적으로 우리는 Brain-Age Prediction problem에 성공적으로 GLD 방법을 적용하였으며, 높은 성능을 달성할 수 있었다. 향후에는 3D ResNet 뿐만 아니라 SFCN, DeepBrainNet과 같은 다양한 모델에 적용하는 연구를 수행할 예정이다.

## Reference
* baseline code : (https://github.com/podismine/BrainAgeReg)
