#!/bin/bash
set -e

#taskset -c 85-112 bash run_compression.sh


# Define the list of methods to loop through
#methods=('iDCT1' 'iDCT2' 'iDCT3' 'iDCT4' 'iDCT5' 'mean' 'bos' 'maxPool' 'pca1' 'pca2' 'rbf1' 'rbf2' 'sigmoid1' 'sigmoid2')
methods=('mean')

# Loop through each method in the list
echo "Processing embeddings: $dir"
for method in "${methods[@]}"
do
    echo 'Running compression for method: ' $method
    # PISCES
    time python scripts/compressing_embeddings.py -e "embeddings/PISCES/PISCES_esm2_8M/" -o "embeddings/PISCES_compressed/PISCES_esm2_8M/" -c $method -l 6
    #time python scripts/compressing_embeddings.py -e "embeddings/PISCES/PISCES_esm2_150M/" -o "embeddings/PISCES_compressed/PISCES_esm2_150M/" -c $method -l 30
    #time python scripts/compressing_embeddings.py -e "embeddings/PISCES/PISCES_esm2_650M/" -o "embeddings/PISCES_compressed/PISCES_esm2_650M/" -c $method -l 33
    #time python scripts/compressing_embeddings.py -e "embeddings/PISCES/PISCES_esm2_15B/" -o "embeddings/PISCES_compressed/PISCES_esm2_15B/" -c $method -l 48
done

