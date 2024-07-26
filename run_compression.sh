#!/bin/bash
set -e

# Define the list of directories to loop through
# excluded due large size
#'HIS7_YEAST_Kondrashov2017'

# directories=('YAP1_HUMAN_Fields2012_singles' 'BRCA1_HUMAN_RING' 'UBE4B_MOUSE_Klevit2013_singles' 
# 'BLAT_ECOLX_Tenaillon2013' 'PABP_YEAST_Fields2013_singles' 'GAL4_YEAST_Shendure2015' 
# 'RL401_YEAST_Bolon2013' 'RL401_YEAST_Fraser2016' 'RL401_YEAST_Bolon2014'  'IF1_ECOLI' 
# 'BRCA1_HUMAN_BRCT'  'TIM_SULSO' 'TIM_THETH'  'TIM_THEMA' 'DLG4_RAT_Ranganathan2012' 
# 'POLG_HCVJF_Sun2014' 'MTH3_HAEAESTABILIZED_Tawfik2015' 'SUMO1_HUMAN_Roth2017'  
# 'CALM1_HUMAN_Roth2017' 'PA_FLU_Sun2015'  'UBC9_HUMAN_Roth2017' 'RASH_HUMAN_Kuriyan'  
# 'TPK1_HUMAN_Roth2017' 'BG_STRSQ_hmmerbit'  'TPMT_HUMAN_Fowler2018' 'PTEN_HUMAN_Fowler2018' 
# 'HSP82_YEAST_Bolon2016' 'BLAT_ECOLX_Ranganathan2015' 'KKA2_KLEPN_Mikkelsen2014' 
# 'BLAT_ECOLX_Palzkill2012' 'BLAT_ECOLX_Ostermeier2014' 'AMIE_PSEAE_Whitehead' 
# 'MK01_HUMAN_Johannessen' 'B3VI55_LIPST_Whitehead2015' 'B3VI55_LIPSTSTABLE'  'parEparD_Laub2015_all' 
# 'HG_FLU_Bloom2016'  'BF520_env_Bloom2018' 'BG505_env_Bloom2018' 'PABP_YEAST_Fields2013_doubles')

directories=('pisces')

# Define the list of methods to loop through
methods=('iDCT' 'mean' 'bos' 'maxPool' 'pca1' 'pca2' 'pca1-2' 'rbf1' 'rbf2' 'sigmoid1' 'sigmoid2')

# Loop through each directory
for dir in "${directories[@]}"
do
    echo "Processing embeddings: $dir"

    # Loop through each method in the list
    for method in "${methods[@]}"
    do
        echo 'Running compression for method: ' $method
        # Run the Python script with the current directory and method as arguments
        #time python scripts/compressing_embeddings.py -e "embeddings/${dir}_esm2_150M/" -c $method
        #time python scripts/compressing_embeddings.py -e "embeddings_pisces/${dir}_esm2_150M/" -c $method -l 30
        time python scripts/compressing_embeddings.py -e "embeddings_pisces/${dir}_esm2_150M/" -c $method -l 30
    done
done
