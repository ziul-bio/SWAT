#!/bin/bash
set -e

# Sequences longer than 1022
DMS_longer=(
    'BRCA1_HUMAN_RING' 'UBE4B_MOUSE_Klevit2013_singles'
    'BRCA1_HUMAN_BRCT' 'POLG_HCVJF_Sun2014'
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
DMS=(
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
    'BLAT_ECOLX_Ostermeier2014' 'BLAT_ECOLX_Ranganathan2015'
    'BLAT_ECOLX_Palzkill2012' 'BLAT_ECOLX_Tenaillon2013'
    'TIM_THEMA' 'B3VI55_LIPSTSTABLE' 'UBC9_HUMAN_Roth2017')



for file in "${DMS_1022[@]}"
do
    echo "Extracting embedding for $file with esm1v_t33_650M_UR90S_1"
    #python scripts/extract_esm2.py ../ESM2_checkpoints/esm1v_t33_650M_UR90S_1.pt "data/DMS_mut_sequences/${file}_muts.fasta" "embeddings/DMS/esm1v_650M/${file}" --repr_layers 33 --include mean &
    #python scripts/extract_esm2.py esm1v_t33_650M_UR90S_1 "data/DMS_mut_sequences/${file}_muts.fasta" "embeddings/DMS/esm1v_650M/${file}" --repr_layers 33 --include mean &

    echo "Extracting embedding for $file with esm2_t6_8M_UR50D"
    python scripts/extract_esm2.py esm2_t6_8M_UR50D "data/DMS_mut_sequences/${file}_muts.fasta" "embeddings/DMS/esm2_8M/${file}" --repr_layers 6 --include mean &

    echo "Extracting embedding for $file with esm2_t12_35M_UR50D"
    python scripts/extract_esm2.py esm2_t12_35M_UR50D "data/DMS_mut_sequences/${file}_muts.fasta" "embeddings/DMS/esm2_35M/${file}" --repr_layers 12 --include mean &

    echo "Extracting embedding for $file with esm2_t30_150M_UR50D"
    python scripts/extract_esm2.py esm2_t30_150M_UR50D "data/DMS_mut_sequences/${file}_muts.fasta" "embeddings/DMS/esm2_150M/${file}" --repr_layers 30 --include bos mean per_tok &

    echo "Extracting embedding for $file with esm2_t33_650M_UR50D"
    python scripts/extract_esm2.py esm2_t33_650M_UR50D "data/DMS_mut_sequences/${file}_muts.fasta" "embeddings/DMS/esm2_650M/${file}" --repr_layers 33 --include mean

    # echo "Extracting embedding for $file with esm2_t36_3B_UR50D"
    # python scripts/extract_esm2.py esm2_t36_3B_UR50D "data/DMS_mut_sequences/${file}_muts.fasta" "embeddings/DMS/esm2_3B/${file}" --repr_layers 36 --include mean

    # echo "Extracting embedding for $file with esm2_t48_15B_UR50D"
    # python scripts/extract_esm2.py esm2_t48_15B_UR50D "data/DMS_mut_sequences/${file}_muts.fasta" "embeddings/DMS/esm2_15B/${file}" --repr_layers 48 --include mean --nogpu
  
    # echo "Extracting embedding for $file with esmc 300M"
    # python scripts/extract_ESMC.py -m esmc-300m -i "data/DMS_mut_sequences/${file}_muts.fasta" -o embeddings/DMS_compressed/esmc_300M/${file}/embed_${file}_mean.pt

    # echo "Extracting embedding for $file with esmc 600M"
    # python scripts/extract_ESMC.py -m esmc-600m -i "data/DMS_mut_sequences/${file}_muts.fasta" -o embeddings/DMS_compressed/esmc_600M/${file}/embed_${file}_mean.pt

    # echo "Extracting embedding for $file with Amplify 120M"
    # python -u scripts/extract_AMPLIFY.py -m models/AMPLIFY_120M -i "data/DMS_mut_sequences/${file}_muts_2.fasta" -o embeddings/DMS_compressed/AMPLIFY_120M/${file}/embed_${file}_mean.pt

    # echo "Extracting embedding for $file with Amplify 350M"
    # python -u scripts/extract_AMPLIFY.py -m models/AMPLIFY_350M -i "data/DMS_mut_sequences/${file}_muts_2.fasta" -o embeddings/DMS_compressed/AMPLIFY_350M/${file}/embed_${file}_mean.pt

done



       
                


        
           



       
               

       
      
        

       
             
           

         
      
        

 
          
           

   
                      
        

 
        

   
          

           
 
