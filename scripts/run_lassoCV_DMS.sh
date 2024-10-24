#!/bin/bash
set -e

# usage
#taskset -c 50-112 bash run_lassoCV.sh 


# DMS all
DMS_all=(
    'YAP1_HUMAN_Fields2012_singles' 'BRCA1_HUMAN_RING'
    'UBE4B_MOUSE_Klevit2013_singles' 'BLAT_ECOLX_Tenaillon2013'
    'PABP_YEAST_Fields2013_singles' 'GAL4_YEAST_Shendure2015'
    'RL401_YEAST_Bolon2013' 'RL401_YEAST_Fraser2016'
    'RL401_YEAST_Bolon2014' 'IF1_ECOLI' 'BRCA1_HUMAN_BRCT'
    'TIM_SULSO' 'TIM_THETH' 'TIM_THEMA' 'DLG4_RAT_Ranganathan2012'
    'POLG_HCVJF_Sun2014' 'MTH3_HAEAESTABILIZED_Tawfik2015'
    'SUMO1_HUMAN_Roth2017' 'CALM1_HUMAN_Roth2017' 'PA_FLU_Sun2015'
    'UBC9_HUMAN_Roth2017' 'RASH_HUMAN_Kuriyan' 'TPK1_HUMAN_Roth2017'
    'BG_STRSQ_hmmerbit' 'TPMT_HUMAN_Fowler2018'
    'PTEN_HUMAN_Fowler2018' 'HSP82_YEAST_Bolon2016'
    'BLAT_ECOLX_Ranganathan2015' 'KKA2_KLEPN_Mikkelsen2014'
    'BLAT_ECOLX_Palzkill2012' 'BLAT_ECOLX_Ostermeier2014'
    'AMIE_PSEAE_Whitehead' 'MK01_HUMAN_Johannessen'
    'B3VI55_LIPST_Whitehead2015' 'B3VI55_LIPSTSTABLE'
    'parEparD_Laub2015_all' 'HG_FLU_Bloom2016' 'BF520_env_Bloom2018'
    'BG505_env_Bloom2018' 'PABP_YEAST_Fields2013_doubles'
)


# DMS 9
DMS9=(
    'BLAT_ECOLX_Ranganathan2015' 'IF1_ECOLI'
    'DLG4_RAT_Ranganathan2012' 'RASH_HUMAN_Kuriyan' 'TIM_THETH'
    'RL401_YEAST_Bolon2013' 'RL401_YEAST_Bolon2014' 
    'PABP_YEAST_Fields2013_singles' 'PABP_YEAST_Fields2013_doubles'
    )



# Define the list of methods to loop through
methods=('mean' 'bos' 'maxPool' 'iDCT1' 'iDCT2' 'iDCT3' 'iDCT4' 'iDCT5' 'pca1' 'pca2' 'rbf1' 'rbf2' 'sigmoid1' 'sigmoid2')
#methods=('mean')

datasets=("${DMS_all[@]}")

# Loop through each dtsectory
for dts in "${datasets[@]}"
do
    echo "Processing embeddings: $dts"
    for method in "${methods[@]}"
    do
        echo 'Running compression for method: ' $method
        python scripts/run_reg_LassoCV.py -i "embeddings/DMS_compressed/${dts}_esm2_150M_compressed/embed_layer_30_${method}.pkl" -m "data/metadata_DMS/${dts}_metadata.csv" -o "results/lassoCV_DMS/esm2_150M_all/${dts}_esm2_150M_${method}.csv"
        
        # python scripts/run_reg_LassoCV.py -i "embeddings/DMS9_compressed/${dts}_esm2_8M/embed_layer_6_${method}.pkl" -m "data/metadata_DMS/${dts}_metadata.csv" -o "results/lassoCV_DMS9/${dts}_esm2_8M_layer_6_${method}.csv"
        # python scripts/run_reg_LassoCV.py -i "embeddings/DMS9_compressed/${dts}_esm2_150M/embed_layer_30_${method}.pkl" -m "data/metadata_DMS/${dts}_metadata.csv" -o "results/lassoCV_DMS9/${dts}_esm2_150M_layer_${method}.csv"
        # python scripts/run_reg_LassoCV.py -i "embeddings/DMS9_compressed/${dts}_esm2_650M/embed_layer_33_${method}.pkl" -m "data/metadata_DMS/${dts}_metadata.csv" -o "results/lassoCV_DMS9/${dts}_esm2_650M_layer_33_${method}.csv"
        # python scripts/run_reg_LassoCV.py -i "embeddings/DMS9_compressed/${dts}_esm2_15B/embed_layer_48_${method}.pkl" -m "data/metadata_DMS/${dts}_metadata.csv" -o "results/lassoCV_DMS9/${dts}_esm2_15B_layer_48_${method}.csv"                                      
    done
done
