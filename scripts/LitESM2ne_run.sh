#!/bin/bash
set -e


#python scripts/LitESM2ne_Lora_trainer.py -i data/DMS_metadata/BLAT_ECOLX_Ranganathan2015_metadata.csv -o "results/fineTune/DMS/LitEsm2neLora" --checkpoint_path ../../wilkelab/pLMs_checkpoints/ESM2/esm2_t36_3B_UR50D.pt

DMS_1022=(
    'DLG4_RAT_Ranganathan2012' 'PA_FLU_Sun2015'
    'HSP82_YEAST_Bolon2016' 'PABP_YEAST_Fields2013_singles' 
    'IF1_ECOLI' 'parEparD_Laub2015_all' 'SUMO1_HUMAN_Roth2017'
    'RL401_YEAST_Fraser2016' 'RL401_YEAST_Bolon2014'
    'RL401_YEAST_Bolon2013' 'CALM1_HUMAN_Roth2017'
    'RASH_HUMAN_Kuriyan' 'TPK1_HUMAN_Roth2017'
    'TPMT_HUMAN_Fowler2018' 'TIM_SULSO' 'TIM_THEMA' 'TIM_THETH'
    'KKA2_KLEPN_Mikkelsen2014' 'BLAT_ECOLX_Tenaillon2013'
    'BLAT_ECOLX_Ranganathan2015' 'BLAT_ECOLX_Ostermeier2014'
    'BLAT_ECOLX_Palzkill2012' 'MTH3_HAEAESTABILIZED_Tawfik2015'
    'AMIE_PSEAE_Whitehead' 'MK01_HUMAN_Johannessen'
    'PTEN_HUMAN_Fowler2018' 'B3VI55_LIPST_Whitehead2015'
    'B3VI55_LIPSTSTABLE' 'BG_STRSQ_hmmerbit'
    'YAP1_HUMAN_Fields2012_singles' 'HG_FLU_Bloom2016'
    )




for file in "${DMS_1022[@]}"
do
    echo "Running regression on Dataset: $file"
    python scripts/LitESM2ne_Lora_trainer.py -i "data/DMS_metadata/${file}_metadata.csv" -o "results/fineTune/DMS/LitEsm2neLora" --checkpoint_path ../../wilkelab/pLMs_checkpoints/ESM2/esm2_t6_8M_UR50D.pt
    python scripts/LitESM2ne_Lora_trainer.py -i "data/DMS_metadata/${file}_metadata.csv" -o "results/fineTune/DMS/LitEsm2neLora" --checkpoint_path ../../wilkelab/pLMs_checkpoints/ESM2/esm2_t12_35M_UR50D.pt
    python scripts/LitESM2ne_Lora_trainer.py -i "data/DMS_metadata/${file}_metadata.csv" -o "results/fineTune/DMS/LitEsm2neLora" --checkpoint_path ../../wilkelab/pLMs_checkpoints/ESM2/esm2_t30_150M_UR50D.pt
    python scripts/LitESM2ne_Lora_trainer.py -i "data/DMS_metadata/${file}_metadata.csv" -o "results/fineTune/DMS/LitEsm2neLora" --checkpoint_path ../../wilkelab/pLMs_checkpoints/ESM2/esm2_t33_650M_UR50D.pt
    python scripts/LitESM2ne_Lora_trainer.py -i "data/DMS_metadata/${file}_metadata.csv" -o "results/fineTune/DMS/LitEsm2neLora" --checkpoint_path ../../wilkelab/pLMs_checkpoints/ESM2/esm2_t36_3B_UR50D.pt
    

done
