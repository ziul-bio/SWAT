[![author](https://img.shields.io/badge/author1-Luiz_Vieira-blue.svg)](https://www.linkedin.com/in/luiz-carlos-vieira-4582797b/) 
[![author](https://img.shields.io/badge/author2-Morgan_Handojo-blue.svg)](https://www.linkedin.com/in/morgan-handojo/) 
[![The Wilke Lab](https://img.shields.io/badge/Wilke-Lab-brightgreen.svg?style=flat)](https://wilkelab.org) 
[![](https://img.shields.io/badge/python-3.8+-yellow.svg)](https://www.python.org/downloads/release/python) 
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/ziul-bio/SWAT/issues) 
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-lightgrey.svg)](http://perso.crans.org/besson/LICENSE.html)


# Bigger is not always better in protein language models
![plot](/figures/fig1_scheme.png)


# About the project:

## Overview

The trend of increasing size in Protein Language Models (pLMs) necessitates the benchmarking of model performance. Due to larger models requiring much more computational power, it is important to save time and resources whenever possible while consistently achieving the most accurate results. While common sense may say that bigger == better, biological complexity and dataset size limitations/diversity can complicate analysis. 

This project offers an in-depth investigation into optimizing transfer learning with Evolutionary Scale Modeling 2 (ESM2) by evaluating the performance of the model's embeddings across different parameter sizes to determine the impact of size on transfer learning in biological datasets. 

### Objective

This project aims to create a deeper understanding of the relationship between model size and maximizing transfer learning using three of ESM2's available parameters sizes: 150M, 650M, and 15B. To investigate this, we benchmarked the perfomance of each of these models on different types of data: highly homogenous Deep Mutational Scanning (DMS) datasets as well as a diverse data from PISCES: A Protein Sequence Culling Server.

### Methodology

* utilized ESM2 (150M, 650M, 15B) to extract protein embeddings from the datasets
* compressed embeddings to reduce dimensionality, with the best method determined to be average pooling
* evaluated predictive performance using LassoCV regression
* statistical analysis performed using a linear mixed-effects model


### Innovation and Impact

By enhancing our understanding of the impact of model size on analysis, we can instead focus on finding new ways to optimize performance by improving smaller models rather than scaling up in size. This will create greater accessibilty for analysis, especially for projects that may not have the resources or computational power to implement larger models.


**Keywords:** ESM2 | pLM Embeddings | Feature compression | Transfer Learning 




# Reproducibility

## Example of how to reproduce our results

```bash
# Once we have all the fasta files and metadata we can extract the embeddings for each fasta.
python scripts/extract.py esm2_t30_150M_UR50D data/DMS_mut_sequences/BLAT_ECOLX_Ostermeier2014_muts.fasta embeddings/DMS/BLAT_ECOLX_Ostermeier2014_esm2_150M --repr_layers 30 --include bos mean per_tok

# Then we can compress the embeddings with the following command
python scripts/compressing_embeddings.py  -e embeddings/DMS/BLAT_ECOLX_Ostermeier2014_esm2_150M -c mean -l 30

# with the compressed embedding we can run the regression model, see script for more details
python scripts/run_reg_Lasso.py -i embeddings/BLAT_ECOLX_Ostermeier2014_esm2_150M_compressed/BLAT_ECOLX_Ostermeier2014_esm2_150M_mean.pkl -m data/metadata_DMS/BLAT_ECOLX_Ostermeier2014_metadata.csv -o results/BLAT_ECOLX_Ostermeier2014_esm2_150M_mean.csv
```


## Using a bash script to loop through all files
Extracting all embeddings on a loop.
```bash
bash run_extraction_DMS.sh

# Compression of the embeddings using a bash script to compress all the embeddings at once
bash run_compression_DMS.sh

# Running the regression model for all the compressed embeddings
bash run_lassoCV_DMS.sh
```

