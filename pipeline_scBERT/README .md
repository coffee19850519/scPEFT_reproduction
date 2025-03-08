### please download the [panglao_human.h5ad](https://mailmissouri-my.sharepoint.com/:u:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/package%20and%20files/panglao_human.h5ad?csf=1&web=1&e=czjXZA)  and [panglao_pretrain.pth](https://mailmissouri-my.sharepoint.com/:u:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/package%20and%20files/panglao_pretrain.pth?csf=1&web=1&e=zvEo7q) into pipeline_scBERT path
### and download the [gene2vec_16906.npy](https://mailmissouri-my.sharepoint.com/:u:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/package%20and%20files/gene2vec_16906.npy?csf=1&web=1&e=2jzybW)  and put it into the performer_pytorch path


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