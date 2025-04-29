[![author](https://img.shields.io/badge/author1-Luiz_Vieira-blue.svg)](https://www.linkedin.com/in/luiz-carlos-vieira-4582797b/) 
[![author](https://img.shields.io/badge/author2-Morgan_Handojo-blue.svg)](https://www.linkedin.com/in/morgan-handojo/) 
[![The Wilke Lab](https://img.shields.io/badge/Wilke-Lab-brightgreen.svg?style=flat)](https://wilkelab.org) 
[![](https://img.shields.io/badge/python-3.8+-yellow.svg)](https://www.python.org/downloads/release/python) 
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-lightgrey.svg)](http://perso.crans.org/besson/LICENSE.html)


# Scaling Down for Efficiency: Medium-Sized Transformer Models for Protein Sequence Transfer Learning
![plot](/figures/fig1_scheme.png)


# About the project:

## Abstract

Protein language models such as the  transformer-based Evolutionary Scale Modeling 2 (ESM2) can offer deep insights into evolutionary and structural properties of proteins. While larger models, such as ESM2 15B, promise to capture more complex patterns in sequence space, they also present practical challenges due to their high dimensionality and high computational cost. We systematically evaluated the performance of all ESM2 models across many biological datasets to determine the impact of model size on transfer learning. Surprisingly, larger models do not always outperform smaller ones, especially when data is limited. Medium sized models, such as ESM2 650M, exhibited consistent performance, falling only slightly behind the 15B parameter model despite being over 20 times smaller. Additionally, we compared various methods of embedding compression to identify the most effective approach, and we found that mean embeddings consistently outperformed other compression methods. Our results show that ESM2 650M with mean embeddings offers an optimal balance between performance and efficiency, making it a practical and scalable choice for transfer learning in a variety of biological applications. 


### Significance

This work challenges the common belief that larger language models always yield better results, here in the context of protein biochemistry. By systematically comparing transformer models of different sizes in transfer learning tasks, we demonstrate that medium size models, such as ESM-2 650M, frequently perform as well as larger variants, specially when data is limited. These findings provide a more efficient strategy for machine learning-based protein analysis and promote the broader accessibility of AI in biology. Smaller, more efficient models can help democratize advanced machine-learning tools, making them more accessible to researchers with limited computational resources


**Keywords:** ESM | Transfer learning | pLM embeddings | Embeddings compression 




# Reproducibility

## Example of how to reproduce our results

0. **Setting up the enviroment**:

```bash
# clone this repository
git clone git@github.com:ziul-bio/SWAT.git

# move in
cd SWAT

# create a python3.10 or higher virtual environment to run ESMC
python3.10 -m venv venv_ESMC

# install our version of the ESMC, modified to garantee reproducibility. See methods.
pip install esm/

# install remaning dependencies
pip install -r requirements.txt


# I highly recoment you to create a separate enviroment to ESM-2.
# since esmc and esm2 have similar libraries names, this may cause conflicts, specially when loading_from_pretrained is called.
# By loading the models from a local folder, I did not encounter any problem.
python3.10 -m venv venv_ESM2

# install remaning dependencies
pip install -r requirements.txt
```

1. **Data pre processing**    
I have made available most of the metadata for both datasets (DMSs and PISCES).
However, due to size limit of 100Mb, per file in the github, you may need to run the pre processing notebook to get the mutated sequences and metadata for the HIS7 dataset. These notebooks, show exactly how I computed and genarated all the target variables for each dataset.
```bash
#Open and run
DMS_pre_processing.ipynb

PISCES_pre_processing.ipynb

```

2. **Extract embeddings**:      
My embeddings directory has approximatlly 1.69 TB in size. At the moment you will have to run the embeddings extraction to be able to run the analysis on your own. In the future I plan to make it avaible online. 

Till then, you can run the following code to extract the embeddings to all DMS and PISCES datasets.   
```bash
# Once we have all the fasta files and metadata we can extract the embeddings for each fasta.

# you can run file by file. Example pisces:
python scripts/extract.py esm2_t30_150M_UR50D data/PISCES/pisces_len64-1022.fasta embeddings/PISCES/esm2_150M --repr_layers 30 --include bos mean per_tok

# or in a loop. Please see the scripts folder. For most of the analysis I have a bash file that will run a loop to all experiments.
bash scripts/run_extraction_DMS.sh
# and
bash scripts/run_compression_PISCES.sh
```

3. **Compress embeddings**:  
```bash
# Then we can compress the embeddings with the following command
python scripts/compressing_embeddings.py -e "embeddings/PISCES/esm2_150M/" -o "embeddings/PISCES_compressed/esm2_150M/" -c mean -l 30
```

4. **Regression Model**:  
```bash
# with the compressed embedding we can run the regression model, see script for more details
python scripts/reg_LassoCV.py -i "embeddings/PISCES_compressed/esm2_150M/embed_pisces_mean.pkl" -m "data/PISCES_metadata/SS_H.csv" -o "results/lassoCV/PISCES/esm2_150M/SS_H_esm2_150M_mean.csv"
```

5. **Fine Tuning**:  
```bash
# All the data and scripts required to fine esm2 model are available in this repositoty.
# I used 4 GPUs on AMD Instinct™ MI100 GPUs, which has 32Gb of memory. Feel free to increase batch size if your machine allow.

## To fine-tune one DMS dataset run:
python scripts/LitESM2ne_Lora_trainer.py -i data/DMS_metadata/BLAT_ECOLX_Tenaillon2013_metadata.csv -o results/fineTune/test --batch_size 2

## To fine-tune all the 31 DMS datasets, as presented in our results, run:
scripts/LitESM2ne_run.sh 
```

#### Footnote    
If you find any issues or need a file or script that I forgot to upload for any reason, please open an issue. I will fix it right away.