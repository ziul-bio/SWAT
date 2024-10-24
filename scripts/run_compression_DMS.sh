#!/bin/bash
set -e

#taskset -c 85-112 bash run_compression.sh

# Define the list of directories to loop through

DMS_all=('YAP1_HUMAN_Fields2012_singles' 'BRCA1_HUMAN_RING' 'UBE4B_MOUSE_Klevit2013_singles' 
'BLAT_ECOLX_Tenaillon2013' 'PABP_YEAST_Fields2013_singles' 'GAL4_YEAST_Shendure2015' 
'RL401_YEAST_Bolon2013' 'RL401_YEAST_Fraser2016' 'RL401_YEAST_Bolon2014' 'IF1_ECOLI' 
'BRCA1_HUMAN_BRCT' 'TIM_SULSO' 'TIM_THETH' 'TIM_THEMA' 'DLG4_RAT_Ranganathan2012' 
'POLG_HCVJF_Sun2014' 'MTH3_HAEAESTABILIZED_Tawfik2015' 'SUMO1_HUMAN_Roth2017'  
'CALM1_HUMAN_Roth2017' 'PA_FLU_Sun2015' 'UBC9_HUMAN_Roth2017' 'RASH_HUMAN_Kuriyan'  
'TPK1_HUMAN_Roth2017' 'BG_STRSQ_hmmerbit' 'TPMT_HUMAN_Fowler2018' 'PTEN_HUMAN_Fowler2018' 
'HSP82_YEAST_Bolon2016' 'BLAT_ECOLX_Ranganathan2015' 'KKA2_KLEPN_Mikkelsen2014' 
'BLAT_ECOLX_Palzkill2012' 'BLAT_ECOLX_Ostermeier2014' 'AMIE_PSEAE_Whitehead' 
'MK01_HUMAN_Johannessen' 'B3VI55_LIPST_Whitehead2015' 'B3VI55_LIPSTSTABLE' 'parEparD_Laub2015_all' 
'HG_FLU_Bloom2016' 'BF520_env_Bloom2018' 'BG505_env_Bloom2018' 'PABP_YEAST_Fields2013_doubles')

DMS9=(
    'BLAT_ECOLX_Ranganathan2015' 'IF1_ECOLI'
    'DLG4_RAT_Ranganathan2012' 'RASH_HUMAN_Kuriyan' 'TIM_THETH'
    'RL401_YEAST_Bolon2013' 'RL401_YEAST_Bolon2014' 
    'PABP_YEAST_Fields2013_singles' 'PABP_YEAST_Fields2013_doubles')



# Define the list of methods to loop through
#methods=('iDCT1' 'iDCT2' 'iDCT3' 'iDCT4' 'iDCT5' 'mean' 'bos' 'maxPool' 'pca1' 'pca2' 'rbf1' 'rbf2' 'sigmoid1' 'sigmoid2')
methods=('mean')

directories=("${DMS9[@]}")

# Loop through each directory
for dir in "${directories[@]}"
do
    echo "Processing embeddings: $dir"

    # Loop through each method in the list
    for method in "${methods[@]}"
    do
        echo 'Running compression for method: ' $method
        # DMS
        time python scripts/compressing_embeddings.py -e "embeddings/DMS9/${dir}_esm2_8M/" -o "embeddings/DMS9_compressed/${dir}_esm2_8M/" -c $method -l 6
        # time python scripts/compressing_embeddings.py -e "embeddings/DMS9/${dir}_esm2_150M/" -o "embeddings/DMS9_compressed/${dir}_esm2_150M/" -c $method -l 30
        # time python scripts/compressing_embeddings.py -e "embeddings/DMS9/${dir}_esm2_650M/" -o "embeddings/DMS9_compressed/${dir}_esm2_650M/" -c $method -l 33
        # time python scripts/compressing_embeddings.py -e "embeddings/DMS9/${dir}_esm2_15B/" -o "embeddings/DMS9_compressed/${dir}_esm2_15B/" -c $method -l 48
                                                             
    done
done
