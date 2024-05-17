#!/bin/bash
set -e

files=('AMIE_PSEAE_Whitehead' 'BRCA1_HUMAN_RING' 'PABP_YEAST_Fields2013_doubles' 'TIM_SULSO' 'B3VI55_LIPSTSTABLE'
'CALM1_HUMAN_Roth2017' 'PABP_YEAST_Fields2013_singles' 'TIM_THEMA' 'B3VI55_LIPST_Whitehead2015' 'DLG4_RAT_Ranganathan2012'
'PA_FLU_Sun2015' 'TIM_THETH' 'BF520_env_Bloom2018' 'GAL4_YEAST_Shendure2015' 'MTH3_HAEAESTABILIZED_Tawfik2015'
'parEparD_Laub2015_all' 'TPK1_HUMAN_Roth2017' 'BG505_env_Bloom2018' 'HG_FLU_Bloom2016' 'POLG_HCVJF_Sun2014'
'TPMT_HUMAN_Fowler2018' 'BG_STRSQ_hmmerbit' 'HIS7_YEAST_Kondrashov2017' 'PTEN_HUMAN_Fowler2018' 'UBC9_HUMAN_Roth2017'
'BLAT_ECOLX_Ostermeier2014' 'HSP82_YEAST_Bolon2016' 'RASH_HUMAN_Kuriyan' 'UBE4B_MOUSE_Klevit2013_singles' 'BLAT_ECOLX_Palzkill2012'
'IF1_ECOLI' 'RL401_YEAST_Bolon2013' 'YAP1_HUMAN_Fields2012_singles' 'BLAT_ECOLX_Ranganathan2015' 'KKA2_KLEPN_Mikkelsen2014'
'RL401_YEAST_Bolon2014' 'BLAT_ECOLX_Tenaillon2013' 'MK01_HUMAN_Johannessen' 'RL401_YEAST_Fraser2016' 'BRCA1_HUMAN_BRCT'
'SUMO1_HUMAN_Roth2017')

for file in "${files[@]}"
do
    echo "Processing embeddings: $file"
    time python scripts/extract.py esm2_t30_150M_UR50D "data/mut_sequences/${file}_muts.fasta" "embeddings/${file}_esm2_150M" --repr_layers 30 --include bos mean per_tok
done







       
                


        
           



       
               

       
      
        

       
             
           

         
      
        

 
          
           

   
                      
        

 
        

   
          

           
 
