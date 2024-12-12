#!/bin/bash
set -e

# usage
#taskset -c 50-112 bash run_lassoCV.sh 

# targets=(
#     'SS_H' 'SS_E' 'SS_Coil' 
#     'PCP_Charge' 'PCP_Hydrophobicity' 'PCP_Instability_index' 'PCP_Isoelectric_point' 'PCP_length' 'PCP_mW_kDa' 
#     'PCP_AAfreq_Ala' 'PCP_AAfreq_Cys' 'PCP_AAfreq_Leu')


targets=(
    'PCP_length' 'PCP_mW_kDa' 
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
        # python scripts/reg_LassoCV.py -i "embeddings/PISCES_compressed/esm1v_650M/embed_pisces_${method}.pkl" -m "data/PISCES_metadata/${target}.csv" -o "results/lassoCV/PISCES/esm1v_650M/${target}_esm1v_650M_${method}.csv"
        # python scripts/reg_LassoCV.py -i "embeddings/PISCES_compressed/esm2_8M/embed_pisces_${method}.pkl" -m "data/PISCES_metadata/${target}.csv" -o "results/lassoCV/PISCES/esm2_8M/${target}_esm2_8M_${method}.csv"
        # python scripts/reg_LassoCV.py -i "embeddings/PISCES_compressed/esm2_35M/embed_pisces_${method}.pkl" -m "data/PISCES_metadata/${target}.csv" -o "results/lassoCV/PISCES/esm2_35M/${target}_esm2_35M_${method}.csv"
        # python scripts/reg_LassoCV.py -i "embeddings/PISCES_compressed/esm2_150M/embed_pisces_${method}.pkl" -m "data/PISCES_metadata/${target}.csv" -o "results/lassoCV/PISCES/${target}_esm2_150M_layer_30_${method}.csv"
        # python scripts/reg_LassoCV.py -i "embeddings/PISCES_compressed/esm2_650M/embed_pisces_${method}.pkl" -m "data/PISCES_metadata/${target}.csv" -o "results/lassoCV/PISCES/${target}_esm2_650M_layer_33_${method}.csv"
        # python scripts/reg_LassoCV.py -i "embeddings/PISCES_compressed/esm2_3B/embed_pisces_${method}.pkl" -m "data/PISCES_metadata/${target}.csv" -o "results/lassoCV/PISCES/esm2_3V/${target}_esm2_3V_${method}.csv"
        # python scripts/reg_LassoCV.py -i "embeddings/PISCES_compressed/esm2_15B/embed_pisces_${method}.pkl" -m "data/PISCES_metadata/${target}.csv" -o "results/lassoCV/PISCES/${target}_esm2_15B_layer_48_${method}.csv"                

        #python scripts/reg_LassoCV.py -i "embeddings/PISCES_compressed/esmc_300M/embed_pisces_${method}.pt" -m "data/PISCES_metadata/${target}.csv" -o "results/lassoCV/PISCES/esmc_300M/${target}_esmc_300M_${method}.csv"                
        #python scripts/reg_LassoCV.py -i "embeddings/PISCES_compressed/esmc_600M/embed_pisces_${method}.pt" -m "data/PISCES_metadata/${target}.csv" -o "results/lassoCV/PISCES/esmc_600M/${target}_esmc_600M_${method}.csv"                
        python scripts/reg_LassoCV.py -i "embeddings/PISCES_compressed/esmc_6B/embed_pisces_${method}.pt" -m "data/PISCES_metadata/${target}.csv" -o "results/lassoCV/PISCES/esmc_6B/${target}_esmc_6B_${method}.csv"                
        
        
        
        echo ' '                      
    done
done