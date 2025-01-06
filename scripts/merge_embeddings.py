#!/usr/bin/env python3 -u

import os
import torch
import numpy as np
import pandas as pd
import pickle
import argparse
import time
from tqdm import tqdm

## Usage
# python scripts/merge_embeddings.py -e embeddings/PISCES/esmc_6B -o embeddings/PISCES_compressed/esmc_6B/embed_pisces_mean.pt -m esmc
# python scripts/merge_embeddings.py -e embeddings/DMS/esmc_6B/PABP_YEAST_Fields2013_doubles/ -o embeddings/DMS_compressed/esmc_6B/PABP_YEAST_Fields2013_doubles/embed_PABP_YEAST_Fields2013_doubles_mean.pt -m esmc

# python scripts/merge_embeddings.py -e embeddings/HIS7/esm2_8M_mean -o embeddings/HIS7_compressed/esm2_8M/embed_pisces_mean.pt -m esm2_8M



def load_embed(base_dir, rep_layer):
    embeddings = {}
    count = 0
    total_files = len(os.listdir(base_dir))
    
    for f in os.listdir(base_dir):
        file = os.path.join(base_dir, f)
        try:
            embed = torch.load(file, weights_only=True)
            label = embed['label']
            tensor = embed['mean_representations'][rep_layer].numpy()
            embeddings[label] = tensor
            
            count += 1
            if count % 5000 == 0:
                print(f"Files processed {count}/{total_files}")
                time.sleep(1)  

        except (EOFError, RuntimeError, ValueError) as e:
            print(f"Corrupted or unreadable file: {file} - Error: {e}")
        except Exception as e:
            print(f"Error loading file {file}: {e}")

    return embeddings


def load_mean_embed_esm2(base_dir, model):
    embeddings = {}
    total_files = len(os.listdir(base_dir))
    files = os.listdir(base_dir)

    layer = {'esm1v_650M':33, 'esm2_8M':6, 'esm2_35M':12, 'esm2_150M':30, 'esm2_650M':33, 'esm2_3B':36, 'esm2_15B':48}

    for f in tqdm(files, total=total_files, desc="Loading mean embeddings", unit="Sequence"):
        #layer = max(torch.load(file, weights_only=True)['mean_representations'].keys())
        file = os.path.join(base_dir, f)
        embed = torch.load(file, weights_only=True)
        label = embed['label']
        tensor = embed['mean_representations'][layer[model]]
        embeddings[label] = tensor
            
    return embeddings


def load_embeds_from_ESMC(base_dir):
    embeddings = {}
    total_files = len(os.listdir(base_dir))
    files = os.listdir(base_dir)

    for f in tqdm(files, total=total_files, desc="Loading mean embeddings", unit="Sequence"):
        file = os.path.join(base_dir, f)
        label = f.split('.')[0]
        embed = torch.load(file, weights_only=True)
        # Very important, I am just taking the mean, because I alreary removed the BOS and EOS token when saving the embeddings
        # Next time, I am planning on keeping the BOS and EOS when downloading the embeddings with the forge
        mean_representations = embed[1:-1, :].mean(0) # I fixed the extraction, SO NOW THE EMBEDDINGS HAVE BOS AND EOS TOKENS, so I need to remove them
        embeddings[label] = mean_representations

    return embeddings


def main(embed_dir, output_file, model):
     
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if model == 'esmc':
        embeds = load_embeds_from_ESMC(embed_dir)
        print(f"Saving embeddings to {output_file}")
        torch.save(embeds, output_file)

    elif 'esm2' in model or 'esm1v' in model:
        embeds = load_mean_embed_esm2(embed_dir, model)
        print(f"Saving embeddings to {output_file}")
        torch.save(embeds, output_file)
       
    else:   
        embeds = load_embed(embed_dir, rep_layer)
        embed_df = pd.DataFrame(embeds).T.reset_index()
        embed_df.rename(columns={'index': 'ID'}, inplace=True)
        embed_df.to_pickle(output_file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--embed_dir', type=str, required=True)
    argparser.add_argument('-o', '--output_file', type=str, required=True)
    argparser.add_argument('-m', '--model', type=str, required=True)
    args = argparser.parse_args()

    model = args.model
    embed_dir = args.embed_dir
    output = args.output_file

    main(embed_dir, output, model)
