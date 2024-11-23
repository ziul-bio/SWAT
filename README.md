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

This work challenges the common belief that larger language models always yield better results, here in the context of protein biochemistry. By systematically comparing transformer models of different sizes in transfer learning tasks, we demonstrate that medium size models, such as ESM2 650M, frequently perform as well as larger variants, specially when data is limited. These findings provide a more efficient strategy for machine learning-based protein analysis and promote the broader accessibility of AI in biology. Smaller, more efficient models can help democratize advanced machine-learning tools, making them more accessible to researchers with limited computational resources


**Keywords:** ESM2 | pLM Embeddings | Feature compression | Transfer Learning 




# Reproducibility

## Example of how to reproduce our results

1. **Extract embeddings**:  
```bash
# Once we have all the fasta files and metadata we can extract the embeddings for each fasta.
python scripts/extract.py esm2_t30_150M_UR50D data/DMS_mut_sequences/BLAT_ECOLX_Ostermeier2014_muts.fasta embeddings/DMS/BLAT_ECOLX_Ostermeier2014_esm2_150M --repr_layers 30 --include bos mean per_tok
```

2. **Compress embeddings**:  
```bash
# Then we can compress the embeddings with the following command
python scripts/compressing_embeddings.py  -e embeddings/DMS/BLAT_ECOLX_Ostermeier2014_esm2_150M -c mean -l 30
```

4. **Regression Model**:  
```bash
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

