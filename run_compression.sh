#!/bin/bash
set -e

# Define the list of directories to loop through
directories=('sumo1_esm2_150M' 'BLAT_ECOLX_Ostermeier2014_esm2_150M'  'pisces_esm2_150M')

# Define the list of methods to loop through
#methods=('mean' 'bos' 'max_pool' 'pca1' 'pca2' 'pca1-2' 'iDCT' 'rbf1' 'rbf2' 'sigmoid1' 'sigmoid2')
methods=('mean' 'bos' 'maxPool' 'pca1' 'pca2' 'pca1-2' 'iDCT')

# Loop through each directory
for dir in "${directories[@]}"
do
    echo "Processing embeddings: $dir"

    # Loop through each method in the list
    for method in "${methods[@]}"
    do
        echo 'Running compression for method: ' $method
        # Run the Python script with the current directory and method as arguments
        time python scripts/compressing_embeddings.py -e "embeddings/${dir}/" -c $method -l 30
    done
done
