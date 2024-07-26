#!/bin/bash
set -e

files=('beta_lactamase' 'subcellular_localization' 'subcellular_localization_2' 'fluorescence' 'stability' 'solubility')


for file in "${files[@]}"
do
    echo "Processing embeddings: $file"
    time python scripts/extract.py esm2_t30_150M_UR50D "data/benchmarks/${file}/${file}_test_data.fasta" "embeddings/embeddings_benchmarks/${file}_test_esm2_150M" --repr_layers 30 --include bos mean per_tok
    time python scripts/extract.py esm2_t30_150M_UR50D "data/benchmarks/${file}/${file}_train_data.fasta" "embeddings/embeddings_benchmarks/${file}_train_esm2_150M" --repr_layers 30 --include bos mean per_tok
    time python scripts/extract.py esm2_t30_150M_UR50D "data/benchmarks/${file}/${file}_valid_data.fasta" "embeddings/embeddings_benchmarks/${file}_valid_esm2_150M" --repr_layers 30 --include bos mean per_tok
done



       
                


        
           



       
               

       
      
        

       
             
           

         
      
        

 
          
           

   
                      
        

 
        

   
          

           
 
