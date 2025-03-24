!/bin/bash
set -e


echo "Extracting embedding for PISCES with esm1v_t33_650M_UR90S_1"
python -u scripts/extract.py models/esm1v_t33_650M_UR90S_1.pt data/PISCES/pisces_len64-1022.fasta embeddings/PISCES_compressed/PISCES_esm1v_650M_mean --repr_layers 33 --include mean

echo "Extracting embedding for PISCES with esm2_t6_8M_UR50D"
python -u scripts/extract.py models/esm2_t6_8M_UR50D.pt data/PISCES/pisces_len64-1022.fasta embeddings/PISCES_compressed/PISCES_esm2_8M_mean --repr_layers 6 --include mean

echo "Extracting embedding for PISCES with esm2_t12_35M_UR50D"
python -u scripts/extract.py models/esm2_t12_35M_UR50D.pt data/PISCES/pisces_len64-1022.fasta embeddings/PISCES_compressed/PISCES_esm2_35M_mean --repr_layers 12 --include mean

echo "Extracting embedding for PISCES with esm2_t30_150M_UR50D"
python -u scripts/extract.py models/esm2_t30_150M_UR50D.pt data/PISCES/pisces_len64-1022.fasta embeddings/PISCES_compressed/PISCES_esm2_150M_mean --repr_layers 30 --include mean

echo "Extracting embedding for PISCES with esm2_t33_650M_UR50D"
python -u scripts/extract.py models/esm2_t33_650M_UR50D.pt data/PISCES/pisces_len64-1022.fasta embeddings/PISCES_compressed/PISCES_esm2_650M_mean --repr_layers 33 --include mean

echo "Extracting embedding for PISCES with esm2_t36_3B_UR50D"
python -u scripts/extract.py models/esm2_t36_3B_UR50D.pt data/PISCES/pisces_len64-1022.fasta embeddings/PISCES_compressed/PISCES_esm2_3B_mean --repr_layers 36 --include mean

echo "Extracting embedding for PISCES with esm2_t48_15B_UR50D"
python -u scripts/extract.py models/esm2_t48_15B_UR50D.pt data/PISCES/pisces_len64-1022.fasta embeddings/PISCES_compressed/PISCES_esm2_15B_mean --repr_layers 48 --include mean



echo "Extracting embedding for PISCES with esm3_300m"
python -u scripts/extract_ESMC.py -m esmc-300m -i data/PISCES/pisces_len64-1022.fasta -o embeddings/PISCES_compressed/PISCES_esmc_300M_mean/pisces_len64-1022.pt

echo "Extracting embedding for PISCES with esm3_600m"
python -u scripts/extract_ESMC.py -m esmc-600m -i data/PISCES/pisces_len64-1022.fasta -o embeddings/PISCES_compressed/PISCES_esmc_600M_mean/pisces_len64-1022.pt




echo "Extracting embedding for PISCES with Amplify 120M"
python -u scripts/extract_AMPLIFY.py -m models/AMPLIFY_120M -i data/PISCES/pisces_len64-1022.fasta -o embeddings/PISCES_compressed/PISCES_AMPLIFY_120M_no_layernorm/pisces_len64-1022.pt

echo "Extracting embedding for PISCES with Amplify 350M"
python -u scripts/extract_AMPLIFY.py -m models/AMPLIFY_350M -i data/PISCES/pisces_len64-1022.fasta -o embeddings/PISCES_compressed/PISCES_AMPLIFY_350M_no_layernorm/pisces_len64-1022.pt
