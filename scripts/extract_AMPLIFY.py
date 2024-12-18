import os
import torch
from torch import nn
import argparse
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# Usage:
#python scripts/extract_AMPLIFY.py -i test/test.fasta -m ../AMPLIFY/amplify_checkpoints/AMPLIFY_120M -o test/embeddings/Blat_mean_embeddings_test3.pt


# load the models locally
def AMPLIFY(model_checkpoint, device):
    model = AutoModel.from_pretrained(model_checkpoint, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    print("Model loaded on:", model.device)
    return model, tokenizer


def extract_mean_representations(model, tokenizer, fasta_file, device):

    mean_representations = {}
    print(f"Reading sequences from {fasta_file} FASTA file")
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    print(f'Number of sequences to process: {len(sequences)}')

    with torch.no_grad():  # Disable gradient calculations

        for seq in tqdm(sequences, desc="Processing sequences", leave=False):
            seq_id = seq.id
            sequence = str(seq.seq)
            seq_length = len(sequence)
            
            # Tokenize with padding and truncation
            tokens = tokenizer(sequence, return_tensors="pt", padding=False, truncation=False)
            tokens = tokens.to(device)
            
            # Get model output
            output = model(tokens['input_ids'], output_hidden_states=True)

            # Get the last hidden state
            embeddings = output.hidden_states[-1][0]  # Last layer, first sequence (batch size = 1)

            # Apply LayerNorm to embeddings
            hidden_size = model.config.hidden_size if hasattr(model, "config") else None
            layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=True).to(device) if hidden_size else None
            if layer_norm:
                embeddings = layer_norm(embeddings)
        
            # Extract the embeddings for the sequence, excluding special tokens
            representations = embeddings[1:seq_length+1, :].detach().to('cpu')  # Excluding [CLS] and padding
            
            # Compute mean representation of the sequence
            mean_representations[seq_id] = representations.mean(dim=0)
    
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
    model, tokenizer = AMPLIFY(model_checkpoint, device)

    # Extract representations
    result = extract_mean_representations(model, tokenizer, path_input_fasta_file, device=device)
    
    # Save results
    torch.save(result, output_file)
    print(f'Process Finished! Results saved to {output_file}')


if __name__ == "__main__":
    main()
