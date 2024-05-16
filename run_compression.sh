#!/bin/bash
set -e

# Define the list of methods to loop through
methods=('mean' 'bos' 'max_pool' 'pca1' 'pca2' pca1-'2' 'rbf1' 'rbf2' 'sigmoid1' 'sigmoid2')

# Loop through each method in the list
for method in "${methods[@]}"
do
    # Run the Python script with the current method as an argument
    echo 'Running compression for method: ' $method
    time python scripts/compressing_embeddings.py  -e embeddings/BLAT_ECOLX_Ostermeier2014_esm2_150M/ -c $method -l 30
    
done