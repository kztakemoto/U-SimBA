# U-SimBA

This repository contains the codes used in our study on [*Simple black-box universal adversarial attacks on medical image classification based on deep neural networks*](https://arxiv.org/abs/2108.04979).

## Terms of use
MIT licensed. Happy if you cite our preprint when using the codes:

Koga K & Takemoto K (2021) Simple black-box universal adversarial attacks on medical image classification based on deep neural networks. arXiv:2108.04979.

## Usage

### 0. Check the usage of U-SimBA with a simple example
See Notebook `U_SimBA_demo.ipynb`.

### 1. Medical images and DNN models
See [hkthirano/MedicalAI-UAP](https://github.com/hkthirano/MedicalAI-UAP) for details, including the requirements.
This repository assumes the directory structure in [hkthirano/MedicalAI-UAP](https://github.com/hkthirano/MedicalAI-UAP).
The images and DNNs (model weights) will stored in the `data` directory.

### 2. Create input and validation data from test dataset
```
python create_data.py --dataset 'melanoma' --nb_samples 1000
python create_data.py --dataset 'oct' --nb_samples 200
python create_data.py --dataset 'chestx' --nb_samples 100
```

### 3. Install U-SimBA
U-SimBA is implemented using implemented using Adversarial Robustness Toolbox (ART, version 1.7.0). Code is available in [our forked version of ART](https://github.com/kztakemoto/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/universal_simba.py).

```
pip install git+https://github.com/kztakemoto/adversarial-robustness-toolbox
```

### 4. Generate UAP
A use example is as follows:

```
python run.py \
--dataset 'melanoma' \
--model_path './data/melanoma/model/inceptionv3.h5' \
--model_type 'InceptionV3' \
--norm_type '2' \
--norm_rate 0.04 \
--epsilon 0.5 \
--freqdim 28 \
--max_iter 5000 \
--nb_sample 1000 \
--targeted -1 \
--gpu '0' \
--save_path ./results/melanoma/nontarget_inceptionv3_L2norm_zeta4_epsilon05_freqdim28_maxiter5000
```

