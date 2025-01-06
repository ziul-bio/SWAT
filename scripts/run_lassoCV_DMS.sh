#!/bin/bash
set -e

# usage
#taskset -c 50-112 bash run_lassoCV.sh 


# Sequences longer than 1022
DMS_longer=(
    'BRCA1_HUMAN_RING' 'UBE4B_MOUSE_Klevit2013_singles'
    'BRCA1_HUMAN_BRCT' 'POLG_HCVJF_Sun2014'
)

# esmc 6B does not accept sequences longer than 2048
DMS_longer2=(
    'BRCA1_HUMAN_RING' 'UBE4B_MOUSE_Klevit2013_singles'
    'BRCA1_HUMAN_BRCT'
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


# DMS
missing_files=('PABP_YEAST_Fields2013_doubles')


# Define the list of methods to loop through
#methods=('bos' 'maxPool' 'iDCT1' 'iDCT2' 'iDCT3' 'iDCT4' 'iDCT5' 'pca1' 'pca2' 'rbf1' 'rbf2' 'sigmoid1' 'sigmoid2')
methods=('mean')

datasets=("${DMS_1022[@]}")

# Loop through each dtsectory
for dts in "${datasets[@]}"
do
    for method in "${methods[@]}"
    do
        echo "Processing dataset: $dts"
        echo 'Running regression for method: ' $method
        
        echo "Esm1v 650M"                                  
        python scripts/reg_LassoCV.py -i "embeddings/DMS_compressed/esm1v_650M/${dts}/embed_layer_33_${method}.pkl" -m "data/DMS_metadata/${dts}_metadata.csv" -o "results/lassoCV/DMS/esm1v_650M/${dts}_${method}.csv"
        
        echo "Esm2 8M"                                  
        python scripts/reg_LassoCV.py -i "embeddings/DMS_compressed/esm2_8M/${dts}/embed_layer_6_${method}.pkl" -m "data/DMS_metadata/${dts}_metadata.csv" -o "results/lassoCV/DMS/esm2_8M/${dts}_${method}.csv"
        
        echo "Esm2 35M"                                  
        python scripts/reg_LassoCV.py -i "embeddings/DMS_compressed/esm2_35M/${dts}/embed_layer_12_${method}.pkl" -m "data/DMS_metadata/${dts}_metadata.csv" -o "results/lassoCV/DMS/esm2_35M/${dts}_${method}.csv"
        
        echo "Esm2 150M"                                  
        python scripts/reg_LassoCV.py -i "embeddings/DMS_compressed/esm2_150M/${dts}/embed_layer_30_${method}.pkl" -m "data/DMS_metadata/${dts}_metadata.csv" -o "results/lassoCV/DMS/esm2_150M/${dts}_${method}.csv"
        
        echo "Esm2 650M"                                  
        python scripts/reg_LassoCV.py -i "embeddings/DMS_compressed/esm2_650M/${dts}/embed_layer_33_${method}.pkl" -m "data/DMS_metadata/${dts}_metadata.csv" -o "results/lassoCV/DMS/esm2_650M/${dts}_${method}.csv"
        
        echo "Esm2 3B"                                  
        python scripts/reg_LassoCV.py -i "embeddings/DMS_compressed/esm2_3B/${dts}/embed_layer_36_${method}.pkl" -m "data/DMS_metadata/${dts}_metadata.csv" -o "results/lassoCV/DMS/esm2_3B/${dts}_${method}.csv"                                      
        
        echo "Esm2 15B"                                  
        python scripts/reg_LassoCV.py -i "embeddings/DMS_compressed/esm2_15B/${dts}/embed_layer_48_${method}.pkl" -m "data/DMS_metadata/${dts}_metadata.csv" -o "results/lassoCV/DMS/esm2_15B/${dts}_${method}.csv"    
        
        echo "Esmc 300M"                                  
        python scripts/reg_LassoCV.py -i "embeddings/DMS_compressed/esmc_300M/${dts}/embed_${dts}_mean.pt" -m "data/DMS_metadata/${dts}_metadata.csv" -o "results/lassoCV/DMS/esmc_300M/${dts}_${method}.csv"    
        
        echo "Esmc 600M"  
        python scripts/reg_LassoCV.py -i "embeddings/DMS_compressed/esmc_600M/${dts}/embed_${dts}_mean.pt" -m "data/DMS_metadata/${dts}_metadata.csv" -o "results/lassoCV/DMS/esmc_600M/${dts}_${method}.csv"    
        
        echo "Esmc 6B"  
        python scripts/reg_LassoCV.py -i "embeddings/DMS_compressed/esmc_6B/${dts}/embed_${dts}_mean.pt" -m "data/DMS_metadata/${dts}_metadata.csv" -o "results/lassoCV/DMS/esmc_6B/${dts}_${method}.csv"    
        
        echo "AMPLIFY 120M"  
        python scripts/reg_LassoCV.py -i "embeddings/DMS_compressed/AMPLIFY_120M/${dts}/embed_${dts}_mean.pt" -m "data/DMS_metadata/${dts}_metadata.csv" -o "results/lassoCV/DMS/amplify_120M/${dts}_${method}.csv"    
        
        echo "AMPLIFY 350M"  
        python scripts/reg_LassoCV.py -i "embeddings/DMS_compressed/AMPLIFY_350M/${dts}/embed_${dts}_mean.pt" -m "data/DMS_metadata/${dts}_metadata.csv" -o "results/lassoCV/DMS/amplify_350M/${dts}_${method}.csv"    
        
        echo " "                                  
    done
done
