#!/bin/bash
set -e

#taskset -c 85-112 bash run_compression.sh

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
    'RL401_YEAST_Bolon2013' 'RL401_YEAST_Fraser2016' 'RL401_YEAST_Bolon2014' 
    'IF1_ECOLI' 'TIM_SULSO' 'TIM_THETH' 'TIM_THEMA' 'DLG4_RAT_Ranganathan2012' 
    'MTH3_HAEAESTABILIZED_Tawfik2015' 'SUMO1_HUMAN_Roth2017'  
    'CALM1_HUMAN_Roth2017' 'PA_FLU_Sun2015' 'UBC9_HUMAN_Roth2017' 'RASH_HUMAN_Kuriyan'  
    'TPK1_HUMAN_Roth2017' 'BG_STRSQ_hmmerbit' 'TPMT_HUMAN_Fowler2018' 'PTEN_HUMAN_Fowler2018' 
    'HSP82_YEAST_Bolon2016' 'BLAT_ECOLX_Ranganathan2015' 'KKA2_KLEPN_Mikkelsen2014' 
    'BLAT_ECOLX_Palzkill2012' 'BLAT_ECOLX_Ostermeier2014' 'AMIE_PSEAE_Whitehead' 
    'MK01_HUMAN_Johannessen' 'B3VI55_LIPST_Whitehead2015' 'B3VI55_LIPSTSTABLE' 'parEparD_Laub2015_all' 
    'HG_FLU_Bloom2016' 'BF520_env_Bloom2018' 'PABP_YEAST_Fields2013_doubles'
    )


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

missing_files=(
    'PABP_YEAST_Fields2013_doubles')



# Define the list of methods to loop through
#methods=('iDCT1' 'iDCT2' 'iDCT3' 'iDCT4' 'iDCT5' 'mean' 'bos' 'maxPool' 'pca1' 'pca2' 'rbf1' 'rbf2' 'sigmoid1' 'sigmoid2')
methods=('mean')

directories=("${missing_files[@]}")

# Loop through each directory
for dir in "${directories[@]}"
do
    echo "Processing embeddings: $dir"

    # Loop through each method in the list
    for method in "${methods[@]}"
    do
        echo 'Running compression for method: ' $method
        # DMS
        # time python scripts/compressing_embeddings.py -e "embeddings/DMS/esm1v_650M/${dir}/" -o "embeddings/DMS_compressed/esm1v_650M/${dir}/" -c $method -l 33
    
        # time python scripts/compressing_embeddings.py -e "embeddings/DMS/esm2_8M/${dir}/" -o "embeddings/DMS_compressed/esm2_8M/${dir}/" -c $method -l 6
        # time python scripts/compressing_embeddings.py -e "embeddings/DMS/esm2_35M/${dir}/" -o "embeddings/DMS_compressed/esm2_35M/${dir}/" -c $method -l 12
        # time python scripts/compressing_embeddings.py -e "embeddings/DMS/esm2_150M/${dir}/" -o "embeddings/DMS_compressed/esm2_150M/${dir}/" -c $method -l 30
        # time python scripts/compressing_embeddings.py -e "embeddings/DMS/esm2_650M/${dir}/" -o "embeddings/DMS_compressed/esm2_650M/${dir}/" -c $method -l 33
        # time python scripts/compressing_embeddings.py -e "embeddings/DMS/esm2_3B/${dir}/" -o "embeddings/DMS_compressed/esm2_3B/${dir}/" -c $method -l 36
        # time python scripts/compressing_embeddings.py -e "embeddings/DMS/esm2_15B/${dir}/" -o "embeddings/DMS_compressed/esm2_15B/${dir}/" -c $method -l 48

    done
done
