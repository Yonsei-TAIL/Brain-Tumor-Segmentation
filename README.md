# Brain Tumor Segmentation

## Getting Started
This repository provides everything necessary to train and evaluate a brain tumor segmentation model.\
The baseline network is [Modified-UNet](https://github.com/pykao/Modified-3D-UNet-Pytorch).

Requirements:
- Python 3 (code has been tested on Python 3.5.6)
- PyTorch (code tested with 1.1.0)
- CUDA and cuDNN (tested with Cuda 10.0)
- Python pacakges : tqdm, opencv-python (4.1), SimpleITK (1.2.0), scipy (1.2.1), imgaug (0.4.0), medpy (0.4.0)

Structure:
- ```data/```: save directory of datasets
- ```datasets/```: data loading code
- ```network/```: network architecture definitions
- ```options/```: argument parser options
- ```utils/```: image processing code, miscellaneous helper functions, and training/evaluation core code
- ```train.py/```: code for model training
- ```test.py/```: code for model evaluation
- ```preprocess_mask.py/```: code for pre-processing masks (refinement of ce mask and peri-tumoral mask generation)

#### Dataset
Out private dataset which has four types of MRI images (FLAIR, T1GD, T1, T2) and three types of mask (necro, ce, T2) divided into train (N=139) and test (N=16) dataset.\
Place the dataset in ```data/``` directory and the dataset architecture must be as below.
```
data
└─── train
│    └─── patientDir001
│    │    │   FLAIR_stripped.nii.gz
│    │    │   T1GD_stripped.nii.gz
│    │    │   T1_stripped.nii.gz
│    │    │   T2_stripped.nii.gz
│    │    │   necro_mask.nii.gz
│    │    │   ce_mask.nii.gz
│    │    │   t2_mask.nii.gz
│    │    │
│    └─── patientDir002
│    └─── patientDir003
│    └─── ...
│
└─── valid
     └─── patientDir00a
     └─── patientDir00b
     └─── ...
```

If the mask doesn't have to be pre-processed (peri-tumoral mask is provided), the patient directory can consist of as below.
```
patientDir
│   FLAIR_stripped.nii.gz
│   T1GD_stripped.nii.gz
│   T1_stripped.nii.gz
│   T2_stripped.nii.gz
│   necro_mask.nii.gz
│   ce_mask.nii.gz
│   peri_mask.nii.gz
```

#### Training and Testing
- Before training, call ```python preprocess_mask.py``` to pre-process masks.\
This python script generates ```ce_refined_mask.nii.gz``` and ```peri_mask.nii.gz``` in each patient directory.

- To train a 3D network, call:
```python train.py --batch_size 1 --in_dim 3 --in_depth 128 --in_res 140```

- Before 2D training, call ```python parsing_2D.py``` to parse 2D datasets, which generates ```2D_slice/``` directory in each patient directory.\
To train a 2D network, call: ```python train.py --batch_size 1 --in_dim 2 --in_res 140```

- To evaluate a network after training, call: ```python evaluate.py --in_dim 2 --resume trained_weights.pth```

- To inference a network, call: ```python inference.py --in_dim 2 --resume trained_weights.pth```\
Note that the data composition must be as below and the generated masks are saved in each patient directory.
```
data
└─── test
│    └─── patientDir001
│    │    │   FLAIR_stripped.nii.gz
│    │    │   T1GD_stripped.nii.gz
│    │    │   T1_stripped.nii.gz
│    │    │   T2_stripped.nii.gz
│    │    │
│    └─── patientDir002
│    └─── patientDir003
│    └─── ...
```

#### Performance (Modified UNet 2D)
|    Metric    | Dice_Necro | Dice_CE | Dice_Peri |
| :----------: | :--------: | :-----: | :-------: |
|     DICE     |   0.7977   |  0.8968 |   0.8147  |
| Hausdorff 95 |   5.4731   |  1.5011 |   5.0860  |
|  Sensitivity |   0.8625   |  0.8754 |   0.9393  |
|  Specificity |   0.9999   |  0.9998 |   0.9977  |

#### Pre-trained Models
- Modified UNet 2D : [Google Drive Link](https://drive.google.com/file/d/19xUNCYensxN_9sxOZ2XanzeD0feTRJ0p/view?usp=sharing)
- Modified UNet 3D : Not released yet.

#### Issue
- In 2D training, the intensity scale of input images is changed by scipy resize function in range [0, 255].\
The resize function should be replaced to keep the pixel value distribution. (Or the mean and std have to be re-calculated.)

## To do list
- [x] Release test code.
- [x] Data Augmentation.
- [x] Inference code which generates mask nifti files.
- [x] Release pre-trained models.
- [X] Additional evaluation metrics (Sensitivity, Specificity, Hausdorff95).