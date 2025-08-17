# Reproduction

This repo is to reproduce results from the manuscript **Harnessing the Power of Single Cell Large Language Models with Parameter Efficient
Fine-Tuning using scPEFT**. Its official repo is [here](https://github.com/coffee19850519/scPEFT). Before reproducing the results, please ensure scPEFT and its dependencies are well installed. See [installation](https://github.com/coffee19850519/scPEFT?tab=readme-ov-file#installation).


## :fire: Update

Here are the added reproductions for the revised manuscript.

### Cell type identification with scFoundation backbone

:soon: Coming soon

### Perturbation prediction with scFoundation backbone

:soon: Coming soon

### Integrate with new PEFT techniques

:soon: Coming soon

### TF identification with Geneformer backbone

The script is `script/gene_cls_clean.py`. To run the script:
```
python -u gene_cls_clean.py \
    --dataset_name "$DATASET_NAME" \
    --output_root "$OUTPUT_PATH" \
    --prompt_type "$PROMPT_TYPE" \
    --ce_weights 7,1
```
| Dataset | Model checkpoints (5 folds) | Data download                                              |            
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| bivalent_promoters   | [Link](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5FGeneformer%5Fcheckpoints%2FTF%20identification%2FBivalent%5Fpromoter&ga=1) |[Link](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5Fdatasets%2FTF%20identification%2FBivalent%5Fpromoter&ga=1)|
|N1_network|[Link](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5FGeneformer%5Fcheckpoints%2FTF%20identification%2FNOTCH1%5Fnetwork&ga=1)|[Link](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5Fdatasets%2FTF%20identification%2FNOTCH1%5Fnetwork&ga=1)|
|tf_regulatory_range|[Link](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5FGeneformer%5Fcheckpoints%2FTF%20identification%2Flong%5Frange%5FTF&ga=1)|[Link](https://mailmissouri-my.sharepoint.com/personal/hefe_umsystem_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhefe%5Fumsystem%5Fedu%2FDocuments%2FscPEFT%5Fdatasets%2FTF%20identification%2Flong%5Frange%5FTF&ga=1)|



### Unsupervised adaptation on a lung cancer atlas

:soon: Coming soon

|Dataset| Model checkpoints                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Atlas download |
|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
|lung cancer atlas| [Encoder(default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EvGr5oWVN-tNndA4wHfP73cBS5U9EMnIz34G2z_UEulJdw?e=rs9Ucf) &#124; [finetune](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EptMG-l_HXJOi7MjZGJqaKgB2tfBuah5-zv2FHpwNt_YQQ?e=widxHA)  &#124; [Prefix](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EnBwESngH_pPho0gM2lu-KUB6Ceyiwv1N96pjSaQX2epBg?e=5sVudA) &#124; [Token](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EkdPADM8ocdFjnclaEthRg4BpwRCVHc0zgXPzcMqMwwh-g?e=xeMpXk) &#124; [LoRA](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/ElpyAj_pTwROotmCFKNJgFEBhcjQrHf60dOzVUXDb5Llvw?e=hRFS63) | [link](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EcD7ObXllz5Mg6bb1eJMpXkBpCVCnc1i0xzfxRZSOyej1g?e=j03QS0)       | 

### Adapter combination

:soon: Coming soon

| Dataset | Model checkpoints                                            | Atlas download                                               |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| NSCLC   | [fold0](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/fold0?csf=1&web=1&e=wCXiWL) \| [fold1](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/fold1?csf=1&web=1&e=gcjgx6) \| [fold2](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/fold2?csf=1&web=1&e=eFGIpx) \| [fold3](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/fold3?csf=1&web=1&e=oxisFY) \| [fold4](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/fold4?csf=1&web=1&e=Vijdfw) | [link](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination?csf=1&web=1&e=RV2HgO) |

### Hyperparameter sensitivity

:soon: Coming soon

————————————————

# Project structure

We adopted the following file structure to organize and link model checkpoints and data with the associated scripts.
To reproduce our results, please ensure that the necessary model checkpoints and datasets have been downloaded from the provided OneDrive links and placed in their corresponding directories.
You can find these download links in the README file within each folder, as well as in the *Get Started* section below.

    ├── checkpoint             // the folder of checkpoint
    │   ├── celltype_identification    
    │       ├── ms
    │           ├── finetune
    │           └── Enocder adapter .....
    │       └── COVID......
    │   └── marker_gene_detection ......
    
    ├── data                 // the folder of dataset
    │    ├── celltype_identification       
    │           ├── ms
    │               ├── 0
                    └── 1 .....
    │    └── marker_gene_detection ......
    
    ├── script               // the scripts and supporting files for downstream tasks with scGPT
    │    ├── Reproduction_Identification.ipynb
    │    ├── Reproduction_MarkerGeneDetection.ipynb
    │    ├── Reproduction_CellPopulationDiscovery.ipynb
    │    ├── Reproduction_BatchCorrection.ipynb
    │    ├── Reproduction_Perturbation.ipynb
    │    └── human_transcription_factors.txt .......
    
    ├──  scgpt              // official scGPT source code
        
    ├──  performer_pytorch  // implementation with scBERT backend
    
    ├──  pipline_pytorch    // the scripts and supporting files with scBERT
    
    ├──  gear_replogle      // perturbation supporting utilities from gear 
    
    ├──  backbone_perturb   // perturbation supporting code with scGPT 
    
    └──  ReadMe.md                  


————————————————
## Get Started

### Cell type identification

| Dataset | Adapter model checkpoints                                                                                                                                                                                                    | 5-fold split processed data                                                                                                                               |
|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| M.S.    | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EnvhW8PFjYRGoTBG3xSmYVgBUeMVtWyzGu8xxJq0mKBlqw?e=dDZPEb) &#124; [Token](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ei8eBH_uvi5Gt_ZLNNo7Yo4Bxf8zuM_VxVCEKVmPSesU1Q?e=9HBjvP) &#124; [LoRA](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Es8j4IHr1XxGjaffooim7_ABu7OmPTnAhqVj84UoQ-7rew?e=tFnIBX) &#124; [Prefix](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EqOR_x5cfRpIuGfggU6VsR0Bi5LLEkFqitzqh9TOkY3edQ?e=1YpKK1) &#124; [finetune](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EmD5Z0cAqglLpNC1pixsr64BPNKSSS7dyeoph4eh13HZCQ?e=kmDKuo) | [M.S.](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EkhxFKs0a5pLlyHIFwPhbAQB3KhJ3t7WeyiaGc7sZxRamw?e=l07pVp) |
| NSCLC   | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ej3SmJ3m6hdJod_K52IQLDABXTNX6MLf_tx4TK2TAZ8_tw?e=tabDrD) &#124; [Token](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EmCSlYzCLuJKnYs7aQKYB-ABg8_M603EhPryQbdzUSkazQ?e=bJKqZg) &#124; [LoRA](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Eh2TKSwBDx5CjrLij7PT_UsBHj1ctPOulMmD_SP_WpB84Q?e=9sHznV) &#124; [Prefix](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EqgNPksAxMZOgTnio807ub8BzmkqVUfaC6b7u-OY-PsAnQ?e=4Flq18) &#124; [finetune](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/ErF2iW0ettlGuVyoPa7XMPQBd2-xRTV8FMGQ9WaMfshK-A?e=Pyuot7)                     | [NSCLC](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EtITDWrq5Q1CoRgpf_2lXY8BfJD7MeM_iPgoBhww_-nB9g?e=dgEghu) |
| COVID   | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EkcBSehxd2hDn-l0F2xdzO0Bx13GJpfANSoe84y-DPrd6Q?e=8vBcH7) &#124; [Token](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EreGBRliDQ9Nhez1R8AlhuYBJeduUNbpqST5_0-mRaNh6A?e=rn8yve) &#124; [LoRA](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/ErDrIimUBHtAiD2cWRQ-lkYBP8aLuw-0F2zKM-qkTTiIdw?e=UQc8t9) &#124; [Prefix](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/ElYGz3WmU-ROtTvGLIiab_IBRgeVGUSspJ9AhwjlBYH0lQ?e=NiayUu) &#124; [finetune](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Etx9d-CLsVpHtltavx2CotMBhgTkN_3uagIzCd1QMJ9P7A?e=mXVeCb)                     | [COVID](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EopHPawZEStOnbyAARakhAsBNSS7wIjjfnCxyCn9RQPdZw?e=GxON87) |




run the Reproduction_Identification.ipynb

```python
key_parameters = dict(
    dataset_name="ms",  # Dataset name  （ms/COVID/NSCLC)
    model_path="../checkpoint/celltype_identification",  # Path to peft model
    data_path="../data/celltype_identification",  # Path to dataset
    peft_type="Encoder_adapter"  # Encoder_adapter/ Token_adapter / Prefix / LoRA / finetune
)
```

run the Reproduction_Identification.py
```
python  Reproduction_Identification.py --dataset_name ms --model_path ../checkpoint/celltype_identification --data_path ../data/celltype_identification --peft_type Encoder_adapter

```


### Cross-species adaptation


| Species | Adapter model checkpoints                                                                                                                                                                                                   | 5-fold split processed data with orthology                                                                                                                                                                                                                                                                                                                                                                                                                     |
|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mouse   | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EtOt35EhuMJElTCeSNhLtOEBMXEO8UaxoL9hYLVRrWrtHw?e=NTHC9y) &#124; [Token](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/En-EUSz5vgFClQB88ptysJgB9uZ8jtfMqw6F8D69h2TT3g?e=SybT8h) &#124; [Prefix](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EmTtG89Hm5xLgLnNTbLdUz4BqZkLYj0OiqemXIoQ79Xvww?e=6herRP) &#124; [LoRA](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EvV5r0f6tShJhNVAXhurEjYBNMTiqKb3fa-imfXCk_BR9A?e=7dLaIg) &#124; [finetune](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EoTdnCWT8L1Dkl5NICrX7rkBezfbox9nwvhy0sc1vecouw?e=DQw0yO)       | [mouse_115746](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/ElbV8m9rWTVAsB52j9ItgBsBMJmu8H18JB3wM5Henze1Yg?e=zCsGN0) &#124;  [mouse_smartseq](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Esm9-Ap9XgZDtqy3uy6hnfMBxwSGbRB5VgOgvAkSw0QEVw?e=Yi6EQa) &#124;  [mouse_10x](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EjS7qzLrj6RLkgDVT6s2yb8BzFlUcoQbbjeuXqk2_l-R2A?e=yoMAdD) |
| Macaques  | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EvXj6q87o_1BgBSFXG2A1HQBZod17cfPp7uhuJA1pyknZw?e=qGdhhF) &#124;  [Token](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Esknm5NWHiJJrm1V31xRUg0Bdp-LV42DuXjxyalCvGErkg?e=qbwpOo) &#124; [Prefix](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EiJ6PMCsKZhDqXTIZuYpU-gBg4VLUt2u024P-PuZ0haxPA?e=fdRpSL) &#124; [LoRA](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EomDYO_p72pAhRAyShMMq5kBjvaUW_2d9oSzBVthiBTCvQ?e=YmyL9r) &#124; [finetune](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/ElIdcr4-Tl5Ev0UidlgzUOIBK2MsarvUCwheLwp7CPa3eA?e=U474ju)                                                                | [link](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EhLh96sk7wpDtqaN75vTcQ0BJJGFQi_SxbavbpogzkgwAg?e=d0E7lz)                                                                                                                                                                                                                                                                                                                        |
| C.elegans   | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Em34lYARKuxLk5zAMGR66GMB1Mp8EXHb6dlyJm1bhvrUOw?e=N5habB) &#124; [Token](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Eur8iHeFzDlCj4q37ABFwmQBXlzXC1IQNSnLmhGGY4XPoA?e=cyyDa5) &#124; [Prefix](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Er4Ogh2ZITVGr1uxZEyaOjIBGuckTee2RBqjyC3aPcLvIA?e=xiHqpi) &#124; [LoRA](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ei0I-x_5Mb5DuXkTu2UBEUwB10EhE2S7WC_zerulkXdz9Q?e=36a2yF) &#124; [finetune](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Em1YLKieiBlCu56EwAP74AwBasYsc_NDIlIcxdkI6BC2Gg?e=nItvat)                                                                      | [link](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EhunvasmYe9IkfqliCwwGCYBTOAKNmNpr8Y1uiWX_4KlYA?e=gdJz3x)                                                                                                                                                                                                                                                                                                                        |


run the Reproduction_Identification.ipynb
```python
key_parameters = dict(
    dataset_name="MergedMonkey",  # Dataset name  （MergedMonkey/mouse_115746/mouse_10x/mouse_smart/elegans）
    model_path="../checkpoint/celltype_identification",  # Path to peft model
    data_path="../data/cross_species",  # Path to dataset
    peft_type="Encoder_adapter"  # Encoder_adapter/ Token_adapter / Prefix / LoRA / finetune
)
```
run the Reproduction_Identification.py
```
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

### Cell Population Discovery

| Dataset            | Adapter model checkpoints                                                                                                                                                                                                    | Processed data                                                                                                                          |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| BMMC & CD34+ cells | [Encoder (default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EqUmBxneFVdElaJ2vmC7YesBhUAh9U_7G0uk_gTbWViKFg?e=GSAEO2)  &#124; [Token](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EulpqLmyKOhIuBnc74_SOKcBymO06ACzOCMFw-Ghxwgu4A?e=lIVkms) &#124; [Prefix](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ek-sR62GGP5It11n56oUKKsBFV_1GmeCAGcZ0Gov9SGfmg?e=h90pbJ) &#124; [LoRA](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EhmKzPXwff9DvYO3z1A9nMQBKyuBG2uLXeloggXXEbnedQ?e=bKFnSg) &#124; [finetune](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EsiAlJr4Kv9NoGc381ntxAgBuFNuVFAkNkfFmvbtaay9Bg?e=ZEyAgW) | [link](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EbSxnn4c6o9Ag0l4Sj_4J_MB0XizQoiAK8Im638BKk7Ang?e=sdGIEp) |


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

