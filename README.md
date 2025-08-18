# Reproduction

This repo is to reproduce results from the manuscript **Harnessing the Power of Single Cell Large Language Models with Parameter Efficient
Fine-Tuning using scPEFT**. Its official repo is [here](https://github.com/coffee19850519/scPEFT). Before reproducing the results, please ensure scPEFT and its dependencies are well installed. See [installation](https://github.com/coffee19850519/scPEFT?tab=readme-ov-file#installation).


## :fire: Update

Here are the added reproductions for the revised manuscript.

### Cell type identification with scFoundation backbone

Run the `script/Reproduction_Identification_scFoundation.py`:
```
python Reproduction_Identification_scFoundation.py \
    --data_path ./data  # Path to dataset \
    --data_name COVID # Dataset name  (ms / COVID / NSCLC) \
    --save_path ./results/ # Path to peft model \
    --use_prompt True  # Wether to use peft methods \
    --peft_type Token_adapter  # Encoder_adapter/ Token_adapter / Prefix / LoRA / finetune \
    --fold_idx 0 # fold index
```

| Dataset | Adapter model checkpoints | Data download link                                                                                                                       |
|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| M.S.    | [Encoder](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EmtpUNjiSZZMhzxdA3A69vABxwBf2fgGz_NHt3jSiwPw7Q?e=hWoSHR) &#124; [Token](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/En3PNppYHg9MigJcTjbCQLkBvj33AArgYuZFvGJn9PpGVw?e=XB6Rss) &#124; [LoRA](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EhO7N5CAWsxLifeS9MqmSp0BJQnAlsucY2XNhJdFTGmG_w?e=a7q3Ob) &#124; [Prefix](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Emt2wqzs-g5PjZg7E5sc_wsBsfC-zfFetnxYwXxXvGPDQQ?e=zIKM0U) &#124; [finetune](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EkRqtG4bm8VCvLA8DKLXEYoBUR4tRszPS8P9jVxfy6d5jw?e=pcFNOK) | [M.S.](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EkhxFKs0a5pLlyHIFwPhbAQB3KhJ3t7WeyiaGc7sZxRamw?e=l07pVp)  |
| NSCLC   | [Encoder](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/ElkPSAZ7R6dNg32y8IPVBWMBF-L0L5USfSz8BNAd0U7pzw?e=hdjV3j) &#124; [Token](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EsFzdA1eQvBNrzQr7R9od9YBT2pLLd6-vvnFg62jcknIZw?e=XahkVn) &#124; [LoRA](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EhzqItaAPtBPgiRTST0sqKgBn_2ihKl4C1ZQcuIs-kmSDQ?e=3WFvw8) &#124; [Prefix](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EvsaaFNLxFlHpJJeforGJWoBwyMaNqjIX3JMMamKdCtvuQ?e=g9cRsp) &#124; [finetune](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ei9E9yZ2-nNMkSrtqShgoCEBxU1gnXFunblyOuJRFepNKA?e=mcTIt1)| [NSCLC](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EtITDWrq5Q1CoRgpf_2lXY8BfJD7MeM_iPgoBhww_-nB9g?e=dgEghu) |
| COVID   | [Encoder](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EsZKwMhC7C5MjhHbLCkl39MBYeqCU1FkosntmGlWq8I7hw?e=7aeBJ6) &#124; [Token](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EoBKOADUGXtLpfYow3etSOUB7b67n-BtlutKeg0jnoJwFw?e=6X0fXP) &#124; [LoRA](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EqtmI_fGleFApqsFyXidlLUBZOufQK_7hXNT9TmUAfXWdQ?e=syKJiU) &#124; [Prefix](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EgWr05Th_ZpKv2uNUR7BQ6ABuG6yX_mhdylkO-OxGk3-Ow?e=OgyIw6) &#124; [finetune](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ep7eyhbLc0tHqGhCVgAoeQ8B1FVp6FSExp4dyuhh1EpqUg?e=48sXPo)| [COVID](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EopHPawZEStOnbyAARakhAsBNSS7wIjjfnCxyCn9RQPdZw?e=GxON87) |


### Perturbation prediction with scFoundation backbone

Run the `script/Reproduction_Perturbation_scFoundation.py`:
```
python Reproduction_Identification_scFoundation.py \
    --data_dir ./data  # Path to dataset \
    --data_name Norman # Dataset name  (Norman / Replogle_k562) \
    --result_dir ./results/ # Path to peft model \
    --peft_type Token_adapter  # Encoder_adapter/ Token_adapter / Prefix / LoRA \
```
| Dataset | Adapter model checkpoints                                                                                                                                                                  | Data download link               |
|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------|
| Norman    | [Encoder](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EgpYhuS5r-pDgtLvwa0YlhwBsGUOO7BNES2V5E91loE0Iw?e=F3vGao) &#124; [Token](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EndIloieCKVItrvL65MYS7UBQnYx0tM5WxkxhtoC1n-uDw?e=hi4xYe) &#124; [LoRA](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/ErQIhQBTISJLjloh-AXYiZgByI9xpgH4ydvsorq2a7g17g?e=oeHWt3) &#124; [Prefix](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EhxlGbBcpflKrM5VoY2j1PsBPdAd2o5DtNXdG2JiQc_B8Q?e=129wpu)| Norman|
| Replogle_k562  | [Encoder](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ej0yMrzuGp9Mm2XV_NbVBToB1ZefQI03v9HW-MHCrmxx2g?e=u1c9Bm) &#124; [Token](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Et-hCAkyoGJBvEVDG3nJKtQBuGiqg2Sgq9GCHY42fOZriw?e=41Awke) &#124; [LoRA](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ej6yu2e44KdFlT_iiS0lCiQB6-iFyq0NESc4l2fgFxSCAg?e=f8FtYu) &#124; [Prefix](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EgBIIobRaKZOtDLeRNgiymgBRcsVxjkimIy7-cR4MNaG_Q?e=agKk85)| Replogle_k562 |

### Integrate with new PEFT techniques

run the Reproduction_Identification.py
```
python  Reproduction_Identification.py --dataset_name ms --model_path ../checkpoint/celltype_identification --data_path ../data/celltype_identification \
        --peft_type HRA # HRA or OFT
```
| Dataset | Adapter model checkpoints                                                                                                         | Data download                                                                                                                            |
|---------|-----------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| M.S.    | [OFT](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EldNH5YI9YpEizbRZsL5i-gB_dLoLotM5FOEpEWDtRZPRA?e=vb3VvX)  &#124; [HRA](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Elixm1uyMtZIn-rwpChdmNIBzB_CCEGAGT05IymxSP50eA?e=xi2dRK)| [M.S.](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EkhxFKs0a5pLlyHIFwPhbAQB3KhJ3t7WeyiaGc7sZxRamw?e=l07pVp)  |
| NSCLC   | [OFT](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Eppsm_DNs3hPnGwI6ZgP6U4BFQkbuS9cO8ubssE2rWSzTw?e=zdf3YV)  &#124; [HRA](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EsCyQ7kcFKtPrVvuU8zF_roBBBPsaxM_4Pn3gWA4EXY2qw?e=NaqRob)| [NSCLC](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EtITDWrq5Q1CoRgpf_2lXY8BfJD7MeM_iPgoBhww_-nB9g?e=dgEghu) |
| COVID   | [OFT](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EikFb1W2iy1Aodv4ndcYUMsBG83SnkD-tDVtnJPjK7T7MQ?e=qHHOgZ)  &#124; [HRA](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EshDV3YADpRCt82LG8PZUBABTcg7ta7C2Lgi9gko-ZJWjw?e=A6Cc62)| [COVID](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EopHPawZEStOnbyAARakhAsBNSS7wIjjfnCxyCn9RQPdZw?e=GxON87) |


### TF identification with Geneformer backbone

The script is `script/gene_cls_clean.py`. To run the script:
```
python -u gene_cls_clean.py \
    --dataset_name "$DATASET_NAME" \   # bivalent_promoters/N1_network/tf_regulatory_range
    --output_root "$OUTPUT_PATH" \
    --prompt_type "$PROMPT_TYPE" \  # encoder_prompt/lora/prefix_prompt/Gene_token_prompt/normal_finetune
    --ce_weights 7,1
```
| Dataset | Model checkpoints (5 folds)                                                                                                                                                                                                               | 5-fold split Data download                                                                                                                                                                                                  |            
| ------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| bivalent_promoters   | Token(default) &#124; Prefix &#124; LoRA &#124; Encoder &#124; Finetune                                                                                                                                                                   | [Link](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EsGaz8SjoS1PoxL_EQkunFQBKejUajFHmaBjVKF18ZJJIg?e=BxIc9g) |
|N1_network| Token(default) &#124; Prefix &#124; LoRA &#124; Encoder &#124; Finetune    | [Link](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EszQVMgBdLdKtK_j2V3OFIMBAdotKUSQyHdUeegOHMsY-w?e=JPrkcs)    |
|tf_regulatory_range| Token(default) &#124; Prefix &#124; LoRA &#124; Encoder &#124; Finetune   | [Link](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Eq0WPoUK_BJLiQjgQZBbPT8BlgTx1P2PIE7GZWgMDgD22A?e=sIAABO)   |



### Unsupervised adaptation on a lung cancer atlas

:soon: Coming soon

|Dataset| Model checkpoints                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Atlas download |
|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
|lung cancer atlas| [Encoder(default)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EvGr5oWVN-tNndA4wHfP73cBS5U9EMnIz34G2z_UEulJdw?e=rs9Ucf) &#124; [finetune](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EptMG-l_HXJOi7MjZGJqaKgB2tfBuah5-zv2FHpwNt_YQQ?e=widxHA)  &#124; [Prefix](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EnBwESngH_pPho0gM2lu-KUB6Ceyiwv1N96pjSaQX2epBg?e=5sVudA) &#124; [Token](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EkdPADM8ocdFjnclaEthRg4BpwRCVHc0zgXPzcMqMwwh-g?e=xeMpXk) &#124; [LoRA](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/ElpyAj_pTwROotmCFKNJgFEBhcjQrHf60dOzVUXDb5Llvw?e=hRFS63) | [link](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EcD7ObXllz5Mg6bb1eJMpXkBpCVCnc1i0xzfxRZSOyej1g?e=j03QS0)       | 

### Adapter combination

The script is `multi_peft_support/tutorial_peft/inference_cell_type.py`. To run the script:

```
# Basic inference with detailed analysis
python inference_cell_type.py \
      --model_dir /mnt/c/Users/wxy/Desktop/esmc/multi_peft_support/all_result/NSCLC \
      --data_base_dir /mnt/c/Users/wxy/Desktop/esmc/multi_peft_support \
      --peft_method peft_NSCLC_TOKEN \
      --fold 0 \
      --pretrained_model ./scGPT_human \
      --batch_size 32 \
      --output_dir ./inference_results \
      --device auto \
      --save_results
```



| Dataset | Model checkpoints                                            | Dataset |
|--------| ------------------------------------------------------------ | ------- |
| Single | [ENCODER]([peft_NSCLC_ENCODER](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/peft_NSCLC_ENCODER?csf=1&web=1&e=niEaDH)) \| [TOKEN]([peft_NSCLC_TOKEN (1)](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/peft_NSCLC_TOKEN (1)?csf=1&web=1&e=gGXTgb) \| [PREFIX]([peft_NSCLC_PREFIX](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/peft_NSCLC_PREFIX?csf=1&web=1&e=bVgICY)) \| [LORA]([peft_NSCLC_LORA](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/peft_NSCLC_LORA?csf=1&web=1&e=txGsdn)) | [NSCLC](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EtITDWrq5Q1CoRgpf_2lXY8BfJD7MeM_iPgoBhww_-nB9g?e=dgEghu) |
| Dual   | [ENCODER_TOKEN]([peft_NSCLC_ENCODER](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/peft_NSCLC_ENCODER?csf=1&web=1&e=niEaDH)) \| [ENCODER_PREFIX]([peft_NSCLC_ENCODER_PREFIX](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/peft_NSCLC_ENCODER_PREFIX?csf=1&web=1&e=3dAtEU)) \| [ENCODER_LORA]([peft_NSCLC_ENCODER_LORA](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/peft_NSCLC_ENCODER_LORA?csf=1&web=1&e=7fNkOr)) \| [TOKEN_PREFIX]([peft_NSCLC_TOKEN_PREFIX](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/peft_NSCLC_TOKEN_PREFIX?csf=1&web=1&e=X3D1to)) \| [TOKEN_LORA]([peft_NSCLC_TOKEN_LORA (1)](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/peft_NSCLC_TOKEN_LORA (1)?csf=1&web=1&e=28fRWU)) \|[PREFIX_LORA]([peft_NSCLC_PREFIX_LORA](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/peft_NSCLC_PREFIX_LORA?csf=1&web=1&e=U6DSwx)) | [NSCLC](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EtITDWrq5Q1CoRgpf_2lXY8BfJD7MeM_iPgoBhww_-nB9g?e=dgEghu) |
| Trio   | [ENCODER_TOKEN_PREFIX]([peft_NSCLC_ENCODER_TOKEN_PREFIX](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/peft_NSCLC_ENCODER_TOKEN_PREFIX?csf=1&web=1&e=ckh9bh)) \|[ENCODER_TOKEN_LORA]([peft_NSCLC_ENCODER_TOKEN_LORA](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/peft_NSCLC_ENCODER_TOKEN_LORA?csf=1&web=1&e=5bvSN4)) \|[ENCODER_PREFIX_LORA]([peft_NSCLC_ENCODER_PREFIX_LORA](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/peft_NSCLC_ENCODER_PREFIX_LORA?csf=1&web=1&e=auD4y1)) \|[TOKEN_PREFIX_LORA]([peft_NSCLC_TOKEN_PREFIX_LORA (1)](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/peft_NSCLC_TOKEN_PREFIX_LORA (1)?csf=1&web=1&e=bFGLjx)) | [NSCLC](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EtITDWrq5Q1CoRgpf_2lXY8BfJD7MeM_iPgoBhww_-nB9g?e=dgEghu) |
| Quad   | [ENCODER_TOKEN_PREFIX_LORA]([peft_NSCLC_ENCODER_TOKEN_PREFIX_LORA](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC/peft_NSCLC_ENCODER_TOKEN_PREFIX_LORA?csf=1&web=1&e=QTfpqg)) | [NSCLC](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EtITDWrq5Q1CoRgpf_2lXY8BfJD7MeM_iPgoBhww_-nB9g?e=dgEghu) |

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

