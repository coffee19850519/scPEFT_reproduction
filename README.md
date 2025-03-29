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
    
    ├──  pyproject.toml
    
    └──  ReadMe.md                  

————————————————
## Data preparation

| Dataset       | Link                                                                                                                                                                             |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| M.S.          | [M.S.](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_datasets/celltype_identification/ms?csf=1&web=1&e=kbwRr3) |
| NSCLC         | [NSCLC](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EtITDWrq5Q1CoRgpf_2lXY8BfJD7MeM_iPgoBhww_-nB9g?e=dgEghu) |
| COVID         | [COVID](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EopHPawZEStOnbyAARakhAsBNSS7wIjjfnCxyCn9RQPdZw?e=GxON87) |
| MergedMonkey  | [MergedMonkey](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EhLh96sk7wpDtqaN75vTcQ0BJJGFQi_SxbavbpogzkgwAg?e=d0E7lz) |
| elegans       | [elegans](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EhunvasmYe9IkfqliCwwGCYBTOAKNmNpr8Y1uiWX_4KlYA?e=gdJz3x) |
| mouse_115746  | [mouse_115746](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/ElbV8m9rWTVAsB52j9ItgBsBMJmu8H18JB3wM5Henze1Yg?e=zCsGN0) |
| mouse_10x     | [mouse_10x](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EjS7qzLrj6RLkgDVT6s2yb8BzFlUcoQbbjeuXqk2_l-R2A?e=yoMAdD) |
| mouse_smart   | [mouse_smart](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Esm9-Ap9XgZDtqy3uy6hnfMBxwSGbRB5VgOgvAkSw0QEVw?e=Yi6EQa) |
| PBMC_10K      | [PBMC_10K](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EaK5MIsz5zFEmvfhEI9ChxkBZ1px0TitJAQtTVArSB1Wqg?e=V3f1Mt) |
| Perirhinal Cortex | [Perirhinal Cortex](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EcQbzOYnm-tJm_IOjy49iW4BqywK8soWQKvAI6xnZQRT3Q?e=V1qNRV) |
| covid_batch | [covid_batch](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EdOsMlIVJL5GjOdtWYkHhaIBtzwqhRnYOP-34dJ8KhQ3oA?e=QWXIFG) |
| BMMC&CD+34Enriched | [BMMC&CD+34Enriched](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EbSxnn4c6o9Ag0l4Sj_4J_MB0XizQoiAK8Im638BKk7Ang?e=8uyepm) |
| adamson       | [adamson](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ei0zoMTWjA9JpAry9j3JzmwBOmvow3jYWMr2PRAeB0U0Lg?e=GK3LCB) |
| norman        | [norman](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Em7ivdCFPbJClM7psJJ10FcBB-dUo215Wfq0_bplXZ8WRQ?e=JGDyjj) |
| Replogle_k562 | [Replogle_k562](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/El1zppwEcQ5DnKC1Z70xHcIBkSwS5Z3bFjYtSazCvMZTqw?e=8v4Lrh) |
| Replogle_rpe1 | [Replogle_rpe1](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Elt8Cilk-jtJgE1OEMBnQfcBOGw8Y12Cut2L9CizgLQBLw?e=poe75X) |

## checkpoint preparation

| Downstream tasks        | Link                                                                                                                                                                                                                                  |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Celltype Identification | [Celltype Identification](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5Fcheckpoints%2Fcelltype%5Fidentification)     |
| MarkerGeneDetection     | [MarkerGeneDetection](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5Fcheckpoints%2Fmarker%5Fgene%5Fdetection)         |
| CellPopulationDiscovery | [CellPopulationDiscovery](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5Fcheckpoints%2Fcell%5Fpopulation%5Fdiscovery) |
| BatchCorrection         | [BatchCorrection](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5Fcheckpoints%2Fbatch%5Fcorrection)                   |
| Perturbation            | [Perturbation](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5Fcheckpoints%2Fperturbation)                     |

## Get Started

### Celltype Identification

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
### BatchCorrection

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

run the Tutorial_Perturbation.ipynb

```python
key_parameters = dict(
    dataset_name="adamson",  # Dataset name
    load_model="../save/perturbation",  # Path to peft model
    data_path="../data/perturbation",  # Path to dataset
    peft_type="Encoder_adapter",  # Encoder_adapter/ Token_adapter / Prefix / LoRA / finetune
)
```

### Cell Population Discovery

run the Tutorial_CellPopulationDiscover.ipynb

```python
key_parameters = dict(
    dataset_name="immune",  # Dataset name
    load_model="../save/cell_population_discovery",  # Path to peft model
    data_path="../data/cell_population_discovery",  # Path to dataset
    peft_type="Encoder_adapter",  # Encoder_adapter/ Token_adapter / Prefix / LoRA / finetune
)
```
