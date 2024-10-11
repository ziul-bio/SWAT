[![author](https://img.shields.io/badge/author-Authors-blue.svg)](https://wilkelab.org) [![](https://img.shields.io/badge/python-3.8+-yellow.svg)](https://www.python.org/downloads/release/python) [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/allmanbrent/picornavirus_2C_protein/issues) [![GPLv3 license](https://img.shields.io/badge/License-GPLv3-lightgrey.svg)](http://perso.crans.org/besson/LICENSE.html)


# Bigger is not always better in protein language models
<sub>*Data Analysis</sub>


**Description:** Data Science, Bioinformatics, Python, bash-unix.


**Links:**

* [Wilke Lab](https://wilkelab.org)


![plot](/fig1_scheme.png)


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
