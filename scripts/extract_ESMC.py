import torch
import argparse
from Bio import SeqIO
from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers


# Usage:
#python extract_ESMC.py -i data/DMS_mut_sequences/BLAT_ECOLX_Tenaillon2013_muts.fasta -m esmc-300m -o Blat_embeddings_test.pt


# load the models locally
def ESMC_300M_202412(model_path: str, device: torch.device | str = "cpu"):
    with torch.device(device):
        model = ESMC(
            d_model=960, n_heads=15, n_layers=30, tokenizer=get_esmc_model_tokenizers()
        ).eval()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    # Convert model parameters to bfloat16
    #model = model.to(torch.bfloat16)
    model = model.to(torch.float32)
    return model


def ESMC_600M_202412(model_path: str, device: torch.device | str = "cpu"):
    with torch.device(device):
        model = ESMC(
            d_model=1152, n_heads=18, n_layers=36, tokenizer=get_esmc_model_tokenizers()
        ).eval()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    # Convert model parameters to float32
    model = model.to(torch.float32)
    return model




class FastaDataLoader:
    """
    Data loader for reading a FASTA file and creating batches based on a token limit.
    
    Args:
    - fasta_file (str): Path to the FASTA file.
    - batch_token_limit (int, optional): Maximum number of tokens per batch. Defaults to 4096.
    - model (object): Model object with a `_tokenize` method for tokenizing sequences.
    """
    def __init__(self, fasta_file, model, batch_token_limit=4096):
        self.fasta_file = fasta_file
        self.batch_token_limit = batch_token_limit
        self.model = model
        self.sequences = list(SeqIO.parse(fasta_file, "fasta"))
        self.total_sequences = len(self.sequences)
        
        # Check for duplicate sequence labels
        sequence_labels = [seq.id for seq in self.sequences]
        assert len(set(sequence_labels)) == len(sequence_labels), "Found duplicate sequence labels"

    def __len__(self):
        # Approximate total number of batches
        total_tokens = sum(len(str(seq.seq)) + 2 for seq in self.sequences)  # +2 for BOS and EOS tokens
        return (total_tokens + self.batch_token_limit - 1) // self.batch_token_limit

    def __iter__(self):
        ids, lengths, seqs = [], [], []
        current_token_count = 0

        for seq in self.sequences:
            seq_length = len(seq.seq)
            token_count = seq_length + 2  # Include BOS and EOS tokens
            if current_token_count + token_count > self.batch_token_limit and ids:
                # Yield current batch if adding the new sequence exceeds the token limit
                tokens = self.model._tokenize(seqs)
                yield ids, lengths, tokens
                ids, lengths, seqs = [], [], []
                current_token_count = 0

            # Add the current sequence to the batch
            ids.append(seq.id)
            lengths.append(seq_length)
            seqs.append(str(seq.seq))
            current_token_count += token_count

        # Yield any remaining sequences
        if ids:
            tokens = self.model._tokenize(seqs)
            yield ids, lengths, tokens



def extract_mean_representations(model, fasta_file):
    mean_representations = {}
    data_loader = FastaDataLoader(fasta_file, model=model)
    
    with torch.no_grad():  # Disable gradient calculations
        for batch_ids, batch_lengths, batch_tokens in tqdm(data_loader, desc="Processing batches", leave=False):
            output = model(batch_tokens)
            logits, embeddings, hiddens = (
                output.sequence_logits,
                output.embeddings,
                output.hidden_states,
            )

            for i, ID in enumerate(batch_ids):
            # Extract the last hidden states for the sequence
                representations =  embeddings[i, 1:batch_lengths[i]+1, :].detach().to('cpu') 
                # compute mean representation of the sequence
                mean_representations[ID] = (representations.mean(dim=0))
    
    return mean_representations


def main():
    parser = argparse.ArgumentParser(description="Extracting ESMC representations from a FASTA file")
    parser.add_argument("-i", "--input_fasta", type=str, required=True, help="Path to the input FASTA file")
    parser.add_argument("-m", "--model_checkpoint", type=str, required=True, help="Model checkpoint identifier")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output file")
    args = parser.parse_args()

    # Define the input parameters
    path_input_fasta_file = args.input_fasta
    model_checkpoint = args.model_checkpoint
    output_file = args.output

    # Create the base directory if it doesn't exist
    base_dir = os.path.dirname(output_file)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model based on the checkpoint identifier
    if model_checkpoint == 'esmc-300m':
        model = ESMC_300M_202412('/stor/work/Wilke/luiz/ESM3_checkpoints/esmc_300m_2024_12_v0.pth', device)
        print("Model transferred to device:", model.device)
    elif model_checkpoint == 'esmc-600m':
        model = ESMC_600M_202412('/stor/work/Wilke/luiz/ESM3_checkpoints/esmc_600m_2024_12_v0.pth', device)
        print("Model transferred to device:", model.device)
    else:
        print("Model not found!")
        print("Choose a valid model checkpoint: 'esmc-300m' or 'esmc-600m'")
        exit(1)

    # Extract representations
    result = extract_mean_representations(model, path_input_fasta_file)
    
    # Save results
    torch.save(result, output_file)
    print(f'Process Finished! Results saved to {output_file}')


if __name__ == "__main__":
    main()
