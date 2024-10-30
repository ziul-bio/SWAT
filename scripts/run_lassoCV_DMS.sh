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


DMS_1022=(
    'YAP1_HUMAN_Fields2012_singles'
    'BLAT_ECOLX_Tenaillon2013' 'PABP_YEAST_Fields2013_singles' 'GAL4_YEAST_Shendure2015' 
    'RL401_YEAST_Bolon2013' 'RL401_YEAST_Fraser2016' 'RL401_YEAST_Bolon2014' 'IF1_ECOLI' 
    'TIM_SULSO' 'TIM_THETH' 'TIM_THEMA' 'DLG4_RAT_Ranganathan2012' 
    'MTH3_HAEAESTABILIZED_Tawfik2015' 'SUMO1_HUMAN_Roth2017'  
    'CALM1_HUMAN_Roth2017' 'PA_FLU_Sun2015' 'UBC9_HUMAN_Roth2017' 'RASH_HUMAN_Kuriyan'  
    'TPK1_HUMAN_Roth2017' 'BG_STRSQ_hmmerbit' 'TPMT_HUMAN_Fowler2018' 'PTEN_HUMAN_Fowler2018' 
    'HSP82_YEAST_Bolon2016' 'BLAT_ECOLX_Ranganathan2015' 'KKA2_KLEPN_Mikkelsen2014' 
    'BLAT_ECOLX_Palzkill2012' 'BLAT_ECOLX_Ostermeier2014' 'AMIE_PSEAE_Whitehead' 
    'MK01_HUMAN_Johannessen' 'B3VI55_LIPST_Whitehead2015' 'B3VI55_LIPSTSTABLE' 'parEparD_Laub2015_all' 
    'HG_FLU_Bloom2016' 'BF520_env_Bloom2018' 'PABP_YEAST_Fields2013_doubles'
    )


# Sequences longer than 1022
DMS_longer=(
    'BRCA1_HUMAN_RING' 'UBE4B_MOUSE_Klevit2013_singles'
    'BRCA1_HUMAN_BRCT' 'POLG_HCVJF_Sun2014' 'BG505_env_Bloom2018'
)


# DMS 9
DMS9=(
    'BLAT_ECOLX_Ranganathan2015' 'IF1_ECOLI'
    'DLG4_RAT_Ranganathan2012' 'RASH_HUMAN_Kuriyan' 'TIM_THETH'
    'RL401_YEAST_Bolon2013' 'RL401_YEAST_Bolon2014' 
    'PABP_YEAST_Fields2013_singles' 'PABP_YEAST_Fields2013_doubles'
)


# Define the list of methods to loop through
#methods=('mean' 'bos' 'maxPool' 'iDCT1' 'iDCT2' 'iDCT3' 'iDCT4' 'iDCT5' 'pca1' 'pca2' 'rbf1' 'rbf2' 'sigmoid1' 'sigmoid2')
methods=('mean')

datasets=("${DMS_longer[@]}")

# Loop through each dtsectory
for dts in "${datasets[@]}"
do
    for method in "${methods[@]}"
    do
        echo "Processing embeddings: $dts"
        echo 'Running compression for method: ' $method
        #echo "Esm1v 650M"                                  
        #python scripts/run_reg_LassoCV.py -i "embeddings/DMS_compressed/${dts}_esm2_150M_compressed/embed_layer_30_${method}.pkl" -m "data/metadata_DMS/${dts}_metadata.csv" -o "results/lassoCV_DMS/esm2_150M_all/${dts}_esm2_150M_${method}.csv"
        
        #echo "Esm2 150M"                                  
        #python scripts/run_reg_LassoCV.py -i "embeddings/DMS_all_compressed/esm1v_650M/${dts}_esm1v_650M/embed_layer_33_${method}.pkl" -m "data/metadata_DMS/${dts}_metadata.csv" -o "results/lassoCV_DMS/esm1v_650M/${dts}_esm1v_650M_${method}.csv"
        
        echo "Esm2 8M"                                  
        python scripts/run_reg_LassoCV.py -i "embeddings/DMS_all_compressed/esm2_8M/${dts}_esm2_8M/embed_layer_6_${method}.pkl" -m "data/metadata_DMS/${dts}_metadata.csv" -o        "results/lassoCV_DMS/esm2_8M/${dts}_esm2_8M_${method}.csv"
        echo "Esm2 35M"                                  
        python scripts/run_reg_LassoCV.py -i "embeddings/DMS_all_compressed/esm2_35M/${dts}_esm2_35M/embed_layer_12_${method}.pkl" -m "data/metadata_DMS/${dts}_metadata.csv" -o     "results/lassoCV_DMS/esm2_35M/${dts}_esm2_35M_${method}.csv"
        echo "Esm2 650M"                                  
        python scripts/run_reg_LassoCV.py -i "embeddings/DMS_all_compressed/esm2_650M/${dts}_esm2_650M/embed_layer_33_${method}.pkl" -m "data/metadata_DMS/${dts}_metadata.csv" -o   "results/lassoCV_DMS/esm2_650M/${dts}_esm2_650M_${method}.csv"
        echo "Esm2 3B"                                  
        python scripts/run_reg_LassoCV.py -i "embeddings/DMS_all_compressed/esm2_3B/${dts}_esm2_3B/embed_layer_36_${method}.pkl" -m "data/metadata_DMS/${dts}_metadata.csv" -o       "results/lassoCV_DMS/esm2_3B/${dts}_esm2_3B_${method}.csv"                                      
        echo "Esm2 15B"                                  
        python scripts/run_reg_LassoCV.py -i "embeddings/DMS_all_compressed/esm2_15B/${dts}_esm2_15B/embed_layer_48_${method}.pkl" -m "data/metadata_DMS/${dts}_metadata.csv" -o     "results/lassoCV_DMS/esm2_15B/${dts}_esm2_15B_${method}.csv"    
        echo " "                                  
    done
done
