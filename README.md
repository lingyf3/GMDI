# GMDI

Code  for the paper Bayesian Domain Adaptation with Gaussian Mixture Domain-Indexing (NeurIPS 2024). 

We implement our model based on the code of [VDI](https://github.com/Wang-ML-Lab/VDI). We appreciate the authors for making their code publicly available.

# Installation

```bash
conda create -n GMDI python=3.8
conda activate GMDI    
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt  
```

# Data 

We have placed all data in ` "data"` folder.

# Quick Start

*  We can train on datasets (Circle, DG_15, DG_60, TPT_48_NS, TPT_48_WE, CompCars) using the instructions below. Taking the Circle dataset as an example:

  ```bash
  cd GMDI/Circle
  python main.py -c config_Circle
  ```

*  Pretrained model is available [here](https://drive.google.com/file/d/1pS6XUndTA6l0g5eJ9z5AZ0T4UgB6R1cD/view?usp=drive_link). Download  the weight and unzip the files. Place the weight in ` "pretrained_weight"` folder. Run the following code to inference. Taking the Circle dataset as an example:

  ```bash
  python inference.py -c config_Circle_inference
  ```

# Reference

This repository contains code from the following work, which should be cited:

```
@inproceedings{VDI,
  title={Domain-Indexing Variational Bayes: Interpretable Domain Index for Domain Adaptation},
  author={Xu, Zihao and Hao, Guang-Yuan and He, Hao and Wang, Hao},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

# Citation

If you find our code useful, please kindly cite the paper:

```
@inproceedings{GMDI,
  title={Bayesian Domain Adaptation with Gaussian Mixture Domain-Indexing},
  author={Ling, Yanfang and Li, Jiyong and Li, Lingbo and Liang, Shangsong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```
