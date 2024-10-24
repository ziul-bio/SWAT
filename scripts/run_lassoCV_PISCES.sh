#!/bin/bash
set -e

# usage
#taskset -c 50-112 bash run_lassoCV.sh 

targets=(
    'SS_H' 'SS_E' 'SS_Coil' 
    'PCP_Charge' 'PCP_Hydrophobicity' 'PCP_Instability_index' 'PCP_Isoelectric_point' 'PCP_length' 'PCP_mW_kDa' 
    'PCP_AAfreq_Ala' 'PCP_AAfreq_Cys' 'PCP_AAfreq_Leu')


# Define the list of methods to loop through
#methods=('iDCT1' 'iDCT2' 'iDCT3' 'iDCT4' 'iDCT5' 'mean' 'bos' 'maxPool' 'pca1' 'pca2' 'rbf1' 'rbf2' 'sigmoid1' 'sigmoid2')
methods=('mean')


for target in "${targets[@]}"
do
    echo "Processing embeddings: $target"
    for method in "${methods[@]}"
    do
        echo 'Running compression for method: ' $method
        python scripts/run_reg_LassoCV.py -i "embeddings/PISCES_compressed/PISCES_esm2_8M/embed_layer_6_${method}.pkl" -m "data/metadata_pisces/metadata_pisces_${target}.csv" -o "results/lassoCV_pisces/${target}_esm2_8M_layer_6_${method}.csv"
        #python scripts/run_reg_LassoCV.py -i "embeddings/PISCES_compressed/PISCES_esm2_150M/embed_layer_30_${method}.pkl" -m "data/metadata_pisces/metadata_pisces_${target}.csv" -o "results/lassoCV_pisces/${target}_esm2_150M_layer_30_${method}.csv"
        #python scripts/run_reg_LassoCV.py -i "embeddings/PISCES_compressed/PISCES_esm2_650M/embed_layer_33_${method}.pkl" -m "data/metadata_pisces/metadata_pisces_${target}.csv" -o "results/lassoCV_pisces/${target}_esm2_650M_layer_33_${method}.csv"
        #python scripts/run_reg_LassoCV.py -i "embeddings/PISCES_compressed/PISCES_esm2_15B/embed_layer_48_${method}.pkl" -m "data/metadata_pisces/metadata_pisces_${target}.csv" -o "results/lassoCV_pisces/${target}_esm2_15B_layer_48_${method}.csv"                                      
    done
done