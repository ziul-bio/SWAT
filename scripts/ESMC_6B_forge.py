import torch
import time
from Bio import SeqIO
from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk.api import ESMProtein, LogitsConfig
from tqdm import tqdm

# Load the token
with open('tokens/ESM3_token.txt', 'r') as f:
    my_token = f.readlines()[0].strip()

# Load the ESM-6B model
# Apply for forge access and get an access token
forge_client = ESM3ForgeInferenceClient(model="esmc-6b-2024-12", url="https://forge.evolutionaryscale.ai", token=my_token)

total_records = len(list(SeqIO.parse("data/PISCES/pisces_len64-1022.fasta", "fasta")))
count = 0
records = SeqIO.parse("data/PISCES/pisces_len64-1022.fasta", "fasta")

for seq_record in tqdm(records, total=total_records):
    count += 1
    
    label = seq_record.id
    sequence = str(seq_record.seq)
  
    protein = ESMProtein(sequence=sequence)
    protein_tensor = forge_client.encode(protein)
    logits_output = forge_client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
    representation = logits_output.embeddings[:, 1:-1, :].detach().cpu().squeeze(0)  # remove the BOS and EOS token
    output_file = f"embeddings/PISCES/PISCES_esmc_6B/{label}.pt"
    torch.save(representation, output_file)

    if count % 10 == 0:
        time.sleep(5)  # Sleep for 5 seconds after processing 100 sequences

