### please download the scBERT pretrained files and checkpoints [panglao_human.h5ad](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EUDkCqqnk2hOiaOl5FSxn5gBQCIbmBDlLDvOmsP41doFWw?e=yIbFaI)  and [panglao_pretrain.pth](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EVesK-hwXoJGq4KNeZ0bewoBiGRjEsdHInv801GL8zBonw?e=UOnNVs) into ***pipeline_scBERT*** path
### and download the scBERT gene tokenizer parameter file [gene2vec_16906.npy](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EVuLpRYVokpKgfppwFG6inEB8IN05BLX5OBmht3v6eir2g?e=CHndIy)  and put it into the ***performer_pytorch*** path


# for scPEFT
```shell
python  cell_classification_scpeft_scBERT.py --data_name NSCLC
python  cell_classification_scpeft_scBERT.py --data_name COVID
python  cell_classification_scpeft_scBERT.py --data_name ms
```


# for finetuned scBERT
```shell
python  cell_classification_finetune_scBERT.py --data_name NSCLC
python  cell_classification_finetune_scBERT.py --data_name COVID
python  cell_classification_finetune_scBERT.py --data_name ms
```

# for native scBERT
```shell
python  Mapping_scBERT.py --data_name NSCLC
python  Mapping_scBERT.py --data_name COVID
python  Mapping_scBERT.py --data_name ms
```