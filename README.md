# scFuMamba
Multi-Omics Fusion with Cross-Modal Knowledge Augmentation for Single-Cell Profil-ing based on Mamba
![Image text](https://github.com/endingcoding/scFuMamba/blob/main/framework.png)

## Prerequisite
* python 3.10.13
* timm 0.4.12
* pytorch 2.0.1
* cudnn 8.7.0
* scanpy 1.10.3
* anndata 0.10.8
* mamba-ssm 1.1.1
* causal-conv1d 1.2.0.post2
* scikit-learn 1.5.1 <br>

## Getting started
If you wish to use your own datasets in **scMMAE**, you need to modify the following six parameters:

- `config.RNA_tokens = config.RNA_component * config.emb_RNA`  
  `RNA_tokens` represents the number of genes used (e.g., 4000 highly variable genes).  
  Note: `config.emb_RNA` must be divisible by the number of attention heads.

- `config.ADT_tokens = config.ADT_component * config.emb_ADT`  
  `ADT_tokens` represents the number of proteins used (e.g., all available proteins).  
  Note: `config.emb_ADT` must also be divisible by the number of attention heads.

  ## Input
The input data is two matrix (RNA: cell_numbers\*1\*gene_numbers, PROTEIN:cell_numbers\*1\*protein_numbers). In addition, input data should be normalized before running the model.

## Example
Use Anaconda to create a Python virtual environment. Here, we will create a Python 3.11 environment named scMMAE
```cmd
conda create -n scFuMamba python=3.10.13
```
Install  packages
```cmd
pip install -r requirements.txt
```
You can run `./scFuMamba/code/scFuMamba_stage1.py`,  and `./scFuMamba/code/scFuMamba_stage2.py` __directly__ as long as you unrar the dataset in the `./scFuMamba/dataset/CITE-seq/*.rar` ,and `./scFuMamba/dataset/RNA-seq/*.rar`.<br>

### ⚙️ Required Dependencies (Mamba-Compatible Versions)

scFuMamba relies on specific versions of Mamba-related libraries. Please manually download and install the following:

- **Mamba (v1.1.1)**  
  🔗 [https://github.com/state-spaces/mamba/releases/tag/v1.1.1](https://github.com/state-spaces/mamba/releases/tag/v1.1.1)

- **causal-conv1d (v1.2.0.post2)**  
  🔗 [https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.2.0.post2](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.2.0.post2)

Make sure these versions are properly installed to ensure compatibility with the Mamba backbone used in scFuMamba.
## Weighted Model
If you need pretrained and fine-tuned model for the dataset in the experiment, please contact [xiaoyj2024@163.com](mailto:xiaoyj2024@163.com)
