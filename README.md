# PITA:  Prompting Task Interaction for Argumentation Mining
This repository implements a prompt tuning model for Argumentation Mining (AM), unifying AM's 3 subtasks in a generative way. 

## Requirements
allennlp==2.8.0
allennlp_models==2.10.1
easydict==1.9
nltk==3.8.1
numpy==1.21.6
pandas==1.3.5
scikit_learn==1.0.2
scipy==1.7.3
tensorboardX==2.5.1
tensorboardX==2.6.2.2
torch==1.9.0+cu111
torch_geometric==2.1.0.post1
tqdm==4.62.3
transformers==4.26.1
ujson==5.5.0

## Preprocess

Please download the original dataset and then use these scripts.

### BART-base


### PE



### CDCP


## Train & Test

```sh
python run_pe7_1.py --config ./configs/pe_bartbase_graph5.json
or
python run_cdcp7_1.py --config ./configs/cdcp_bartbase_graph5.json
```

### Reproducibility

We experiment on one Nvidia A100(40G) with CUDA version $11.1$. 
All parameters are in json files, i.e. `configs/pe_bartbase_graph5.json` and `configs/cdcp_bartbase_graph5.json` .


# Citation

