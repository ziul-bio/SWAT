import os
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

# esmc_6B only supports seq_len<=2048
#datasets = ['BRCA1_HUMAN_RING', 'BRCA1_HUMAN_BRCT', 'UBE4B_MOUSE_Klevit2013_singles']


datasets = {
    1: ['BF520_env_Bloom2018'], # complete
    2: ['MTH3_HAEAESTABILIZED_Tawfik2015', 'SUMO1_HUMAN_Roth2017', 'TPMT_HUMAN_Fowler2018', 'TPK1_HUMAN_Roth2017', 'IF1_ECOLI'], # complete
    3: ['BLAT_ECOLX_Palzkill2012', 'BLAT_ECOLX_Tenaillon2013', 'YAP1_HUMAN_Fields2012_singles', 'BLAT_ECOLX_Ostermeier2014'], # complete
    4: ['GAL4_YEAST_Shendure2015', 'PABP_YEAST_Fields2013_singles', 'B3VI55_LIPSTSTABLE', 'CALM1_HUMAN_Roth2017'], # complete
    5: ['B3VI55_LIPST_Whitehead2015', 'TIM_THETH', 'RL401_YEAST_Bolon2013', 'DLG4_RAT_Ranganathan2012'], # complete
    6: ['HSP82_YEAST_Bolon2016', 'TIM_THEMA', 'RASH_HUMAN_Kuriyan', 'UBC9_HUMAN_Roth2017'], # complete
    7: ['PA_FLU_Sun2015', 'AMIE_PSEAE_Whitehead', 'RL401_YEAST_Bolon2014'], # complete
    8: ['RL401_YEAST_Fraser2016', 'MK01_HUMAN_Johannessen'], # complete
    9: ['BG_STRSQ_hmmerbit', 'BLAT_ECOLX_Ranganathan2015'], # complete
    10: ['PTEN_HUMAN_Fowler2018', 'KKA2_KLEPN_Mikkelsen2014'], # complete
    11: ['TIM_SULSO', 'parEparD_Laub2015_all'], # complete
    12: ['BG505_env_Bloom2018'], 
    13: ['HG_FLU_Bloom2016'],
    14: ['PABP_YEAST_Fields2013_doubles'],
    }


for dataset in datasets[12]:
    print(f'Reading the input file: {dataset}')
    base_dir = 'data/DMS_mut_sequences'
    input_file = f"{base_dir}/{dataset}_muts.fasta"
    output_dir = f"embeddings/DMS/esmc_6B/{dataset}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


    total_records = len(list(SeqIO.parse(input_file, "fasta")))
    print(f'Total records: {total_records}')
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
        #representation = logits_output.embeddings[:, 1:-1, :].detach().cpu().squeeze(0)  # remove the BOS and EOS token
        representation = logits_output.embeddings.detach().cpu().squeeze(0)  # Keeping BOS and EOS token
        output_file = f"{output_dir}/{label}.pt"
        torch.save(representation, output_file)

        if count % batch_size == 0: # Pause every 50 files
            elapsed_time = time.time() - start_time 
            if elapsed_time < interval: # Pause if the elapsed time is less than 60 seconds
                time.sleep(interval - elapsed_time) #sleep the remaining time

            start_time = time.time()


