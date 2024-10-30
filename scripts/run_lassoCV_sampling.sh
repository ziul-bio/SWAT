#!/bin/bash
set -e

# usage
#taskset -c 50-112 bash run_lassoCV_sampling.sh 

models=('esm1v_650M' 'esm2_8M' 'esm2_35M' 'esm2_150M' 'esm2_650M' 'esm2_3B' 'esm2_15B')
method='mean'

for model in "${models[@]}" 
do
    echo "Running regression on embeddings with LassoCV and sampling for model $model"
    
    # echo "BLAT ECOLX 2015"
    # time python scripts/run_reg_LassoCV_sampling.py -i "embeddings/BLAT_2015_compressed/${model}/BLAT_ECOLX_2015_${model}_${method}_df.pkl" -m "data/metadata_DMS/BLAT_ECOLX_Ranganathan2015_metadata.csv" -o "results/lassoCV_BLAT2015/lassoCV_sampling/BLAT_ECOLX_2015_sampling_${model}_${method}.csv"
    
    # echo "PABP doubles"
    # time python scripts/run_reg_LassoCV_sampling.py -i "embeddings/PABP_doubles_compressed/${model}/PABP_doubles_${model}_${method}_df.pkl" -m "data/metadata_DMS/PABP_YEAST_Fields2013_doubles_metadata.csv" -o "results/lassoCV_PABP_doubles/lassoCV_sampling/PABP_doubles_sampling_${model}_${method}.csv"
    
    # echo "parEparD all"
    time python scripts/run_reg_LassoCV_sampling.py -i "embeddings/parEparD_all_compressed/${model}/parEparD_all_${model}_${method}_df.pkl" -m "data/metadata_DMS/parEparD_Laub2015_all_metadata.csv" -o "results/lassoCV_parEparD_all/lassoCV_sampling/parEparD_all_sampling_${model}_${method}.csv"
                                                                                                                                                                           
    # echo "HIS7"
    # time python scripts/run_reg_LassoCV_sampling.py -i "embeddings/HIS7_compressed/HIS7_${model}/HIS7_${model}_${method}_df.pkl" -m "data/metadata_DMS/HIS7_YEAST_Kondrashov2017_metadata.csv" -o "results/lassoCV_HIS7/lassoCV_sampling/HIS7_YEAST_sampling_${model}_${method}.csv"
done
