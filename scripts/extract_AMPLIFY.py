import torch
from torch import nn
import argparse
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# Usage:
#python extract_AMPLIFY.py -i data/DMS_mut_sequences/BLAT_ECOLX_Tenaillon2013_muts.fasta -m AMPLIFY_120M -o test/Blat_embeddings_test2.pt


# load the models locally
def AMPLIFY(model_checkpoint, device):
    model = AutoModel.from_pretrained(model_checkpoint, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    model = model.to(device)
    print("Model loaded on:", model.device)
    return model, tokenizer


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class FastaDataLoader:
    """
    Data loader for reading a FASTA file and returning batches of sequence IDs, lengths, and tokenized sequences.
    Designed to work with tqdm.
    
    Args:
    - fasta_file (str): Path to the FASTA file.
    - batch_size (int): Number of sequences per batch.
    - model (object): Model object with a `_tokenize` method for tokenizing sequences.
    """
    def __init__(self, fasta_file, batch_size, tokenizer):
        self.fasta_file = fasta_file
        self.batch_size = batch_size
        self.tokenizer=tokenizer
        self.sequences = list(SeqIO.parse(fasta_file, "fasta"))
        self.total = len(self.sequences)

    def __len__(self):
        return (self.total + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        ids, lengths, seqs = [], [], []
        for seq in self.sequences:
            ids.append(seq.id)
            lengths.append(len(seq.seq))
            seqs.append(str(seq.seq))
            
            if len(ids) == self.batch_size:
                tokens = self.tokenizer(seqs, return_tensors="pt", padding=True, truncation=False)
                tokens = tokens['input_ids']
                yield ids, lengths, tokens
                ids, lengths, seqs = [], [], []

        # Yield any remaining sequences if they don't fill the last batch
        if ids:
            tokens = self.tokenizer(seqs, return_tensors="pt", padding=True, truncation=False)
            tokens = tokens['input_ids']
            #tokens = tokens['input_ids']
            yield ids, lengths, tokens


def extract_mean_representations3(model, tokenizer, fasta_file, batch_size=2, device="cpu"):
    """
    Extracts mean representations for sequences in a FASTA file using the last hidden layer
    of a model, applying LayerNorm or RMSNorm to the last hidden layer.

    Args:
        model: The model to extract embeddings from.
        tokenizer: The tokenizer to process the sequences.
        fasta_file: Path to the FASTA file.
        batch_size: Number of sequences per batch.
        device: Device to run computations on (e.g., "cpu" or "cuda").

    Returns:
        A dictionary with sequence IDs as keys and their LayerNorm mean representations.
    """
    mean_representations = {}
    data_loader = FastaDataLoader(fasta_file, batch_size=batch_size, tokenizer=tokenizer)
    
    # Determine the hidden size from the model
    hidden_size = model.config.hidden_size if hasattr(model, "config") else None
    
    # Initialize LayerNorm and move to correct device
    layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=True).to(device) if hidden_size else None
    
    for batch_ids, batch_lengths, batch_tokens in tqdm(data_loader, desc="Processing batches", leave=False):
        batch_tokens = batch_tokens.to(device)  # Ensure tokens are on the correct device
        output = model(batch_tokens, output_hidden_states=True)
      
        embeddings = output.hidden_states[-1]  # Extract the last hidden layer
        
        # Apply LayerNorm if initialized
        if layer_norm:
            embeddings = layer_norm(embeddings)

        for i, ID in enumerate(batch_ids):
            # Extract the normalized last hidden states for the sequence
            representations = embeddings[i, 1:batch_lengths[i] + 1, :].detach().to('cpu') 
            # Compute the mean representation of the sequence
            mean_representations[ID] = representations.mean(dim=0)
    
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
    if model_checkpoint == 'AMPLIFY_350M':
        model, tokenizer = AMPLIFY('/stor/work/Wilke/luiz/AMPLIFY/amplify_checkpoints/AMPLIFY_120M', device)
       
    if model_checkpoint == 'AMPLIFY_350M':
        model, tokenizer = AMPLIFY('/stor/work/Wilke/luiz/AMPLIFY/amplify_checkpoints/AMPLIFY_350M', device)
      
    else:
        print("Model not found!")
        print("Choose a valid model checkpoint: 'AMPLIFY_120M' or 'AMPLIFY_350M'")
        exit(1)

    # Extract representations
    result = extract_mean_representations(model, tokenizer, path_input_fasta_file, batch_size=4, device=device)
    
    # Save results
    torch.save(result, output_file)
    print(f'Process Finished! Results saved to {output_file}')


if __name__ == "__main__":
    main()
