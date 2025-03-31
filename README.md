# Reproduction

This repo is to reproduce results from the manuscript **Harnessing the Power of Single Cell Large Language Models with Parameter Efficient
Fine-Tuning using scPEFT**. Its official repo is [here](https://github.com/coffee19850519/scPEFT). Before reproducing the results, please ensure scPEFT and its dependencies are well installed. See [installation](https://github.com/coffee19850519/scPEFT?tab=readme-ov-file#installation).

# Project structure

    ├── checkpoint             // the path of checkpoint
    │   ├── celltype_identification    
    │       ├── ms
    │           ├── finetune
    │           └── Enocder adapter .....
    │       └── COVID......
    │   └── marker_gene_detection ......
    
    ├── data                 // the path of dataset
    │    ├── celltype_identification       
    │           ├── ms
    │               ├── 0
                    └── 1 .....
    │    └── marker_gene_detection ......
    
    ├── script              
    │    ├── Reproduction_Identification.ipynb
    │    ├── Reproduction_MarkerGeneDetection.ipynb
    │    ├── Reproduction_CellPopulationDiscovery.ipynb
    │    ├── Reproduction_BatchCorrection.ipynb
    │    ├── Reproduction_Perturbation.ipynb
    │    └── human_transcription_factors.txt .......
    
    ├──  scgpt      

    └──  ReadMe.md                  

————————————————
## Get Started

### Cell type Identification

| Dataset | Adapter model checkpoints                                                                                                                            | 5-fold split processed data                                                                                                                               |
|---------|------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| M.S.    | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EnvhW8PFjYRGoTBG3xSmYVgBUeMVtWyzGu8xxJq0mKBlqw?e=dDZPEb) | [M.S.](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_datasets/celltype_identification/ms?csf=1&web=1&e=kbwRr3) |
| NSCLC   | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ej3SmJ3m6hdJod_K52IQLDABXTNX6MLf_tx4TK2TAZ8_tw?e=tabDrD) | [NSCLC](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EtITDWrq5Q1CoRgpf_2lXY8BfJD7MeM_iPgoBhww_-nB9g?e=dgEghu) |
| COVID   | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EkcBSehxd2hDn-l0F2xdzO0Bx13GJpfANSoe84y-DPrd6Q?e=8vBcH7) | [COVID](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EopHPawZEStOnbyAARakhAsBNSS7wIjjfnCxyCn9RQPdZw?e=GxON87) |




run the Reproduction_Identification.ipynb

```python
key_parameters = dict(
    dataset_name="ms",  # Dataset name  （ms/COVID/NSCLC)
    model_path="../checkpoint/celltype_identification",  # Path to peft model
    data_path="../data/celltype_identification",  # Path to dataset
    peft_type="Encoder_adapter"  # Encoder_adapter/ Token_adapter / Prefix / LoRA / finetune
)

key_parameters = dict(
    dataset_name="MergedMonkey",  # Dataset name  （MergedMonkey/mouse_115746/mouse_10x/mouse_smart/elegans）
    model_path="../checkpoint/celltype_identification",  # Path to peft model
    data_path="../data/cross_species",  # Path to dataset
    peft_type="Encoder_adapter"  # Encoder_adapter/ Token_adapter / Prefix / LoRA / finetune
)
```
run the Reproduction_Identification.py
```
python  Reproduction_Identification.py --dataset_name ms --model_path ../checkpoint/celltype_identification --data_path ../data/celltype_identification --peft_type Encoder_adapter
python  Reproduction_Identification.py --dataset_name MergedMonkey --model_path ../checkpoint/celltype_identification --data_path ../data/cross_species --peft_type Encoder_adapter

```


### Cross-species adaptation


| Species | Adapter model checkpoints                                                                                                                            | 5-fold split processed data                                                                                                                                                                                                                                                                                                                                                                                                                        |
|---------|------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mouse   | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EtOt35EhuMJElTCeSNhLtOEBMXEO8UaxoL9hYLVRrWrtHw?e=NTHC9y) | [mouse_115746](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/ElbV8m9rWTVAsB52j9ItgBsBMJmu8H18JB3wM5Henze1Yg?e=zCsGN0)   [mouse_smartseq](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Esm9-Ap9XgZDtqy3uy6hnfMBxwSGbRB5VgOgvAkSw0QEVw?e=Yi6EQa)   [mouse_10x](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EjS7qzLrj6RLkgDVT6s2yb8BzFlUcoQbbjeuXqk2_l-R2A?e=yoMAdD) |
| Monkey  | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EvXj6q87o_1BgBSFXG2A1HQBZod17cfPp7uhuJA1pyknZw?e=qGdhhF) | [link](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EhLh96sk7wpDtqaN75vTcQ0BJJGFQi_SxbavbpogzkgwAg?e=d0E7lz)                                                                                                                                                                                                                                                                                                            |
| C.elegans   | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Em34lYARKuxLk5zAMGR66GMB1Mp8EXHb6dlyJm1bhvrUOw?e=N5habB) | [link](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EhunvasmYe9IkfqliCwwGCYBTOAKNmNpr8Y1uiWX_4KlYA?e=gdJz3x)                                                                                                                                                                                                                                                                                                            |


### MarkerGeneDetection

run the Reproduction_MarkerGeneDetection.ipynb

```python
key_parameters = dict(
    dataset_name="COVID",  # Dataset name
    model_path="../checkpoint/marker_gene_detection",  # Path to peft model
    data_path="../data/marker_gene_detection",  # Path to dataset
    peft_type="Encoder_adapter"  # Encoder_adapter/ Token_adapter / Prefix / LoRA / finetune
)
```
run the Reproduction_MarkerGeneDetection.py
```
python  Reproduction_MarkerGeneDetection.py --dataset_name COVID --model_path ../checkpoint/marker_gene_detection --data_path ../data/marker_gene_detection --peft_type Encoder_adapter
```

### Cell Population Discovery

| Dataset            | Adapter model checkpoints                                                                                                                             | Processed data                                                                                                                          |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| BMMC & CD34+ cells | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EqUmBxneFVdElaJ2vmC7YesBhUAh9U_7G0uk_gTbWViKFg?e=GSAEO2)  | [link](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EbSxnn4c6o9Ag0l4Sj_4J_MB0XizQoiAK8Im638BKk7Ang?e=sdGIEp) |


run the Tutorial_CellPopulationDiscover.ipynb

```python
key_parameters = dict(
    dataset_name="immune",  # Dataset name
    load_model="../save/cell_population_discovery",  # Path to peft model
    data_path="../data/cell_population_discovery",  # Path to dataset
    peft_type="Encoder_adapter",  # Encoder_adapter/ Token_adapter / Prefix / LoRA / finetune
)
```


### BatchCorrection


| Dataset           | Adapter model checkpoints                                                                                                                            | Processed data                                                                                                                          |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| PBMC_10K          | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/ElChtb4eJqVLuEjMeVcRDCgBRN7X5g8R0kfjU649Te-mZw?e=5fmD7l) | [link](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EaK5MIsz5zFEmvfhEI9ChxkBZ1px0TitJAQtTVArSB1Wqg?e=V3f1Mt) |
| Perirhinal Cortex | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Euz1O0aWnQhDrvu2o7sohwMBcoBmjw8tP0OAR5S_mRRE-w?e=s7tesB) | [link](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EcQbzOYnm-tJm_IOjy49iW4BqywK8soWQKvAI6xnZQRT3Q?e=V1qNRV) |
| COVID_batch       | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EnJ2XSvbubBDm0vvR-iv7t0B9rHSHlaRdjfuq0gM7nicug?e=x79zAo) | [link](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EdOsMlIVJL5GjOdtWYkHhaIBtzwqhRnYOP-34dJ8KhQ3oA?e=QWXIFG) |

run the Tutorial_BatchCorrection.ipynb

```python
key_parameters = dict(
    dataset_name="PBMC_10K",  # Dataset name
    load_model="../save/batch_correction",  # Path to peft model
    data_path="../data/batch_correction",  # Path to dataset
    peft_type="Encoder_adapter"  # Encoder_adapter/ Token_adapter / Prefix / LoRA / finetune
)
```

### Perturbation


| Dataset       | Adapter model checkpoints                                                                                                                            | Processed data                                                                                                                             |
|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| Adamson       | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/ErX8q3eZATFMiKJquRbXd_kBV73WLReO7pbgtJh1zXlFsw?e=X0br4x)                                                                                                                                                 | [Adamson](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ei0zoMTWjA9JpAry9j3JzmwBOmvow3jYWMr2PRAeB0U0Lg?e=GK3LCB) |
| Norman        | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EqcNjgOOhJ1DqvSDCx7XD14BrA4HsaSl-FQa_0aPnPUTZA?e=t6n1bT) | [link](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Em7ivdCFPbJClM7psJJ10FcBB-dUo215Wfq0_bplXZ8WRQ?e=JGDyjj)    |
| Replogle_k562 | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EsUwYknZbSNNilGNXtDctP0BmUDgB9MvDe5wIfU8FWHTmg?e=DCD5Ep) | [link](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/El1zppwEcQ5DnKC1Z70xHcIBkSwS5Z3bFjYtSazCvMZTqw?e=8v4Lrh)    |
| Replogle_rpe1 | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EgfTx6fwM-dCusykllNmnVgBpHAI4RCDWztgy3B41lLOxQ?e=992A7p) | [link](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Elt8Cilk-jtJgE1OEMBnQfcBOGw8Y12Cut2L9CizgLQBLw?e=poe75X)    |


run the Tutorial_Perturbation.ipynb

```python
key_parameters = dict(
    dataset_name="adamson",  # Dataset name
    load_model="../save/perturbation",  # Path to peft model
    data_path="../data/perturbation",  # Path to dataset
    peft_type="Encoder_adapter",  # Encoder_adapter/ Token_adapter / Prefix / LoRA / finetune
)
```

