#!/usr/bin/env python3 -u
import os
import torch
import shutil
import numpy as np
import pandas as pd

import pickle
import argparse
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import MinMaxScaler
from scipy.fftpack import dct, idct
from sklearn.manifold import TSNE

# python scripts/compressing_embeddings.py  -e embeddings/sumo1_esm2_150M/ -c mean -l 30

def features_scaler(features):
    '''Scale the features by min-max scaler, to ensure that the features selected by Lasso are not biased by the scale of the features.
    Also, the features are scaled across the rows, i.e., the features are scaled across the sequence length.'''
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_features = scaler.fit_transform(features)
    return pd.DataFrame(scaled_features)

def pca_transformation(embeddings, num_pca_components=2):
    '''Transform the embeddings using PCA.
    First I transposed the embeddings, because PCA works on the features, not on the samples.
    So my features will be the sequence length and the samples will be the model dimensions.
    After PCA, I transposed the embeddings back to obtain a single vector of the transformed embeddings,
    each row will represented by a pca of model dimentsions'''
    features = features_scaler(embeddings).T
    pca = PCA(num_pca_components)
    embed_trans = pca.fit_transform(features)
    return embed_trans.T

def kernel_pca_rbf_transformation(embeddings, num_pca_components=2):
    '''Transform the embeddings using kernel PCA'''
    features = features_scaler(embeddings).T
    kpca_rbf = KernelPCA(kernel="rbf", gamma=None, n_components=num_pca_components, n_jobs=48)
    kpca_rbf_features = kpca_rbf.fit_transform(features)
    return kpca_rbf_features.T

def kernel_pca_sigmoid_transformation(embeddings, num_pca_components=2):
    '''Transform the embeddings using kernel PCA'''
    features = features_scaler(embeddings).T
    kpca_sigmoid = KernelPCA(kernel="sigmoid", gamma=None, n_components=num_pca_components, n_jobs=48)
    kpca_sigmoid_features = kpca_sigmoid.fit_transform(features)
    return kpca_sigmoid_features.T

def tSNE_transformation(embeddings, num_pca_components=2):
    '''Transform the embeddings using tSNE'''
    features = features_scaler(embeddings).T
    tSNE_model = TSNE(n_components=num_pca_components, n_jobs=48, metric='cosine')
    tSNE_features = tSNE_model.fit_transform(features)
    return tSNE_features.T


# original implementation at https://github.com/MesihK/prost/blob/master/src/pyprost/prosttools.py
def iDCTquant(v,n):
    f = dct(v.T, type=2, norm='ortho')
    trans = idct(f[:,:n], type=2, norm='ortho')
    for i in range(len(trans)):
        trans[i] = scale(trans[i])
    return trans.T

def scale(v):
    M = np.max(v)
    m = np.min(v)
    return (v - m) / float(M - m)

#def quant2D(emb,n=5,m=44): #v01 original implementation
def quant2D(emb,n=8,m=80):
    dct = iDCTquant(emb[1:len(emb)-1],n)
    ddct = iDCTquant(dct.T,m).T
    ddct = ddct.reshape(n*m) # turn a 2D array into a 1D vector
    return (ddct*127).astype('int8')
    


####### load embeddings functions #######    
def load_per_tok_embeds(embed_dir, compression_method, rep_layer):
    count=0
    for file in os.listdir(embed_dir):
        total_files = len(os.listdir(embed_dir))
        tmp_folder = 'tmp/'
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)
        
        if file.endswith('.pt'):
            file_path = os.path.join(embed_dir, file)
            label = file.split('.pt')[0]

            if compression_method == 'mean':
                embed = np.array(torch.load(file_path, weights_only=True)['mean_representations'][rep_layer])
                embedding = embed
                np.save(f"{tmp_folder}/{label}.npy", embedding)
            elif compression_method == 'bos':
                embed = np.array(torch.load(file_path, weights_only=True)['bos_representations'][rep_layer])
                embedding = embed
                np.save(f"{tmp_folder}/{label}.npy", embedding)
            elif compression_method == 'maxPool':
                embed = np.array(torch.load(file_path, weights_only=True)['representations'][rep_layer])
                embed_trans = embed.max(axis=0)
                embedding = embed_trans
                np.save(f"{tmp_folder}/{label}.npy", embedding)
            elif compression_method == 'pca1':
                embed = np.array(torch.load(file_path, weights_only=True)['representations'][rep_layer])
                embed_trans = pca_transformation(embed)
                embedding = embed_trans[0]
                np.save(f"{tmp_folder}/{label}.npy", embedding)
            elif compression_method == 'pca2':
                embed = np.array(torch.load(file_path, weights_only=True)['representations'][rep_layer])
                embed_trans = pca_transformation(embed)
                embedding = embed_trans[1]
                np.save(f"{tmp_folder}/{label}.npy", embedding)
            elif compression_method == 'pca1-2':
                embed = np.array(torch.load(file_path, weights_only=True)['representations'][rep_layer])
                embed_trans = pca_transformation(embed)
                embedding = embed_trans.reshape(-1)
                np.save(f"{tmp_folder}/{label}.npy", embedding)
            elif compression_method == 'rbf1':
                embed = np.array(torch.load(file_path, weights_only=True)['representations'][rep_layer])
                embed_trans = kernel_pca_rbf_transformation(embed)
                embedding = embed_trans[0]
                np.save(f"{tmp_folder}/{label}.npy", embedding)
            elif compression_method == 'rbf2':
                embed = np.array(torch.load(file_path, weights_only=True)['representations'][rep_layer])
                embed_trans = kernel_pca_rbf_transformation(embed)
                embedding = embed_trans[1]
                np.save(f"{tmp_folder}/{label}.npy", embedding)
            elif compression_method == 'sigmoid1':
                embed = np.array(torch.load(file_path, weights_only=True)['representations'][rep_layer])
                embed_trans = kernel_pca_sigmoid_transformation(embed)
                embedding = embed_trans[0]
                np.save(f"{tmp_folder}/{label}.npy", embedding)
            elif compression_method == 'sigmoid2':
                embed = np.array(torch.load(file_path, weights_only=True)['representations'][rep_layer])
                embed_trans = kernel_pca_sigmoid_transformation(embed)
                embedding = embed_trans[1]
                np.save(f"{tmp_folder}/{label}.npy", embedding)
            elif compression_method == 'iDCT1': #v01 original implementation
                embed = np.array(torch.load(file_path, weights_only=True)['representations'][rep_layer])
                embedding = quant2D(embed, n=5, m=44)
                np.save(f"{tmp_folder}/{label}.npy", embedding)
            elif compression_method == 'iDCT2': #v02 (n=10, m=64), final dimention 640 same as esm2_150M
                embed = np.array(torch.load(file_path, weights_only=True)['representations'][rep_layer])
                embedding = quant2D(embed, n=10, m=64) 
                np.save(f"{tmp_folder}/{label}.npy", embedding)
            elif compression_method == 'iDCT3': #v03 (n=10, m=128), final dimention 1280 same as esm2_650M 
                embed = np.array(torch.load(file_path, weights_only=True)['representations'][rep_layer])
                embedding = quant2D(embed, n=10, m=128) 
                np.save(f"{tmp_folder}/{label}.npy", embedding)
            elif compression_method == 'iDCT4': #v04 (n=10, m=512), final dimention 5120 same as esm2_15B
                embed = np.array(torch.load(file_path, weights_only=True)['representations'][rep_layer])
                embedding = quant2D(embed, n=10, m=512)
                np.save(f"{tmp_folder}/{label}.npy", embedding)
            elif compression_method == 'iDCT5': #v05 (n=10, m=640), final dimention 6400
                embed = np.array(torch.load(file_path, weights_only=True)['representations'][rep_layer])
                embedding = quant2D(embed, n=10, m=640)
                np.save(f"{tmp_folder}/{label}.npy", embedding)
            else:
                raise ValueError('Invalid compression method')
                print('Valid compression methods are: mean, bos, max_pool, pca1, pca2, pca1-2, rbf1, rbf2, sigmoid1, sigmoid2')
          
        count+=1
        if count % 1000 == 0:
            print(f'{count}/{total_files} files compressed')

    embeddings = {}
    for file in os.listdir(tmp_folder):
        if file.endswith('.npy'):
            file_path = os.path.join(tmp_folder, file)
            label = file.split('.npy')[0]
            embeddings[label] = np.load(file_path)

    # Remove the temporary folder and its contents
    shutil.rmtree(tmp_folder)

    return embeddings



def main(embed_dir, out_dir, compression_method, rep_layer):
    print(f"Compressing embeddings from {embed_dir}")
    compressed_embed = load_per_tok_embeds(embed_dir, compression_method, rep_layer)
    l = str(rep_layer)
    c = str(compression_method)
    out_dir = str(out_dir)
    print(f"Saving compressed embeddings to {out_dir}")
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(f'{out_dir}/embed_layer_{l}_{c}.pkl', 'wb') as f:
        pickle.dump(compressed_embed, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compress embeddings given a transformation method')
    parser.add_argument('-e', '--embed_dir', type=str, help='')
    parser.add_argument('-o', '--out_dir', type=str, help='')
    parser.add_argument('-c', '--compression_method', type=str, help='')
    parser.add_argument('-l', '--layer', type=int, default=30, help='')
    
    args = parser.parse_args()
    embed_dir = args.embed_dir
    out_dir = args.out_dir
    compression_method = args.compression_method
    layer = args.layer

    main(embed_dir, out_dir, compression_method, layer)