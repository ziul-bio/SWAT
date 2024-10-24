#!/bin/bash
set -e

# usage
#taskset -c 50-112 bash run_lassoCV_sampling.sh 

models=('esm2_650M' 'esm2_15B')
method='mean'

for model in "${models[@]}" 
do
    echo "Running regression on embeddings with LassoCV and sampling for model $model"
    time python scripts/run_reg_LassoCV_sampling.py -i "embeddings/HIS7_compressed/HIS7_${model}_${method}.csv" -m "data/metadata_DMS/HIS7_YEAST_Kondrashov2017_metadata.csv" -o "results/lassoCV_HIS7/HIS7_YEAST_Kondrashov2017_sampling_${model}_${method}.csv"
done
