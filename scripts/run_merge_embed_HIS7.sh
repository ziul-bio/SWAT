#!/bin/bash
set -e

#taskset -m 85-112 bash run_compression.sh

echo 'Running compression for model: ' $model

python scripts/merge_embeddings.py -e embeddings/HIS7/esm1v_650M_mean/ -o embeddings/HIS7_compressed/esm1v_650M/embed_HIS7_mean.pt -m esm1v_650M

#python scripts/merge_embeddings.py -e embeddings/HIS7/esm2_8M_mean/ -o embeddings/HIS7_compressed/esm2_8M/embed_HIS7_mean.pt -m esm2_8M
python scripts/merge_embeddings.py -e embeddings/HIS7/esm2_35M_mean/ -o embeddings/HIS7_compressed/esm2_35M/embed_HIS7_mean.pt -m esm2_35M
python scripts/merge_embeddings.py -e embeddings/HIS7/esm2_150M_mean/ -o embeddings/HIS7_compressed/esm2_150M/embed_HIS7_mean.pt -m esm2_150M
python scripts/merge_embeddings.py -e embeddings/HIS7/esm2_650M_mean/ -o embeddings/HIS7_compressed/esm2_650M/embed_HIS7_mean.pt -m esm2_650M
python scripts/merge_embeddings.py -e embeddings/HIS7/esm2_3B_mean/ -o embeddings/HIS7_compressed/esm2_3B/embed_HIS7_mean.pt -m esm2_3B
python scripts/merge_embeddings.py -e embeddings/HIS7/esm2_15B_mean/ -o embeddings/HIS7_compressed/esm2_15B/embed_HIS7_mean.pt -m esm2_15B

#python scripts/merge_embeddings.py -e embeddings/PISCES/esmc_6B -o embeddings/PISCES_compressed/esmc_6B/embed_pisces_mean.pt -m esmc_6B