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
print("Loading the ESM-6B model")
forge_client = ESM3ForgeInferenceClient(model="esmc-6b-2024-12", url="https://forge.evolutionaryscale.ai", token=my_token)

print('Reading the input file')
input_file = "data/PISCES/pisces_len64-1022_left.fasta"
output_dir = "embeddings/PISCES/esmc_6B"
total_records = len(list(SeqIO.parse(input_file, "fasta")))
records = SeqIO.parse(input_file, "fasta")
count = 0

# I can only process 50 files per minute
start_time = time.time()
batch_size = 50
interval = 60



for seq_record in tqdm(records, total=total_records):
    count += 1
    
    label = seq_record.id
    sequence = str(seq_record.seq)
  
    protein = ESMProtein(sequence=sequence)
    protein_tensor = forge_client.encode(protein)
    logits_output = forge_client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
    # try to not remove the BOS and EOS token next time
    representation = logits_output.embeddings[:, 1:-1, :].detach().cpu().squeeze(0)  # remove the BOS and EOS token
    output_file = f"{output_dir}/{label}.pt"
    torch.save(representation, output_file)

    if count % batch_size == 0: # Pause every 50 files
        elapsed_time = time.time() - start_time 
        if elapsed_time < interval: # Pause if the elapsed time is less than 60 seconds
            time.sleep(interval - elapsed_time) #sleep the remaining time

        start_time = time.time()


