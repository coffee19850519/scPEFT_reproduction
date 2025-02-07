# Reproduce

Scripts to reproduce results obtained in the manuscript

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

## Requirements

1. Navigate to the project directory and create a conda environment:
    ```shell
    cd scPEFT
    conda env create -f environment.yaml
    ```
2. Activate the conda environment:
    ```shell
    conda activate scGPT
    ```

## Data preparation

| Dataset       | Link                                                                                                                                                                             |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| M.S.          | [M.S.](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5Fdatasets%2Fcelltype%5Fidentification%2Fms) |
| NSCLC         | [NSCLC](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5Fdatasets%2Fcelltype%5Fidentification%2FNSCLC) |
| COVID         | [COVID](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5Fdatasets%2Fcelltype%5Fidentification%2FCOVID) |
| MergedMonkey  | [MergedMonkey](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5Fdatasets%2Fcross%5Fspecies%2FMergedMonkey) |
| elegans       | [elegans](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5Fdatasets%2Fcross%5Fspecies%2Felegans) |
| mouse_115746  | [mouse_115746](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5Fdatasets%2Fcross%5Fspecies%2Fmouse%5F115746) |
| mouse_10x     | [mouse_10x](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5Fdatasets%2Fcross%5Fspecies%2Fmouse%5F10x) |
| mouse_smart   | [mouse_smart](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5Fdatasets%2Fcross%5Fspecies%2Fmouse%5Fsmart) |
| PBMC_10K      | [PBMC_10K](https://mailmissouri-my.sharepoint.com/:u:/r/personal/hefe_umsystem_edu/Documents/scPEFT_datasets/batch_correction/PBMC_10K.h5ad?csf=1&web=1&e=S7rEMp) |
| Perirhinal Cortex | [Perirhinal Cortex](https://mailmissouri-my.sharepoint.com/:u:/r/personal/hefe_umsystem_edu/Documents/scPEFT_datasets/batch_correction/Perirhinal%20Cortex.h5ad?csf=1&web=1&e=uJmNcg) |
| covid_batch | [covid_batch](https://mailmissouri-my.sharepoint.com/:u:/r/personal/hefe_umsystem_edu/Documents/scPEFT_datasets/batch_correction/covid_subsampled.h5ad?csf=1&web=1&e=XwYAZJ) |
| BMMC&CD+34Enriched | [BMMC&CD+34Enriched](https://mailmissouri-my.sharepoint.com/:u:/r/personal/hefe_umsystem_edu/Documents/scPEFT_datasets/cell_population_discovery/immune.h5ad?csf=1&web=1&e=cUcHrK) |
| adamson       | [adamson](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_datasets/perturbation/adamson?csf=1&web=1&e=PzTR8l) |
| norman        | [norman](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_datasets/perturbation/norman?csf=1&web=1&e=IEJ3ZF) |
| Replogle_k562 | [Replogle_k562](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/perturbation/replogle_k562_essential?csf=1&web=1&e=crrxnh) |
| Replogle_rpe1 | [Replogle_rpe1](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/perturbation/replogle_rpe1_essential?csf=1&web=1&e=hlvQs9) |

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
