### please download the panglao_human.h5ad and panglao_pretrain.pth into pipeline_scBERT path from the link
### and download the gene2vec_16906.npy from the link and put it into the performer_pytorch path


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