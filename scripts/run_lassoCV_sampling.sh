#!/bin/bash
set -e

# usage
#taskset -c 50-112 bash run_lassoCV_sampling.sh 

#models=('esm1v_650M' 'esm2_8M' 'esm2_35M' 'esm2_150M' 'esm2_650M' 'esm2_3B' 'esm2_15B')
models=('esmc_300M' 'esmc_600M')

for model in "${models[@]}" 
do
    echo "Running regression on embeddings with LassoCV and sampling for model $model"
    
    # echo "BLAT ECOLX 2015"
    # time python scripts/reg_LassoCV_sampling.py -i "embeddings/BLAT_2015_compressed/${model}/BLAT_ECOLX_2015_${model}_${method}_df.pkl" -m "data/metadata_DMS/BLAT_ECOLX_Ranganathan2015_metadata.csv" -o "results/lassoCV_BLAT2015/lassoCV_sampling/BLAT_ECOLX_2015_sampling_${model}_${method}.csv"
    
    # echo "PABP doubles"
    # time python scripts/reg_LassoCV_sampling.py -i "embeddings/PABP_doubles_compressed/${model}/PABP_doubles_${model}_${method}_df.pkl" -m "data/metadata_DMS/PABP_YEAST_Fields2013_doubles_metadata.csv" -o "results/lassoCV_PABP_doubles/lassoCV_sampling/PABP_doubles_sampling_${model}_${method}.csv"
                                                                                                                                                                           
    # echo "HIS7"
    time python scripts/reg_LassoCV_sampling.py -i "embeddings/HIS7_compressed/${model}/embed_HIS7_mean.pt" -m "data/HIS7_metadata/HIS7_YEAST_Kondrashov2017_metadata.csv" -o "results/lassoCV_sampling/HIS7/HIS7_sampling_${model}_mean.csv"
done
