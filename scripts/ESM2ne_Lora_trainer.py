import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from torch.optim import AdamW

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from sklearn import metrics
from scipy.stats import spearmanr

from peft import get_peft_model, LoraConfig
from ESM2ne_Lora_regression import Load_from_pretrained



#python scripts/ESM2ne_Lora_trainer.py -i data/DMS_metadata/YAP1_HUMAN_Fields2012_singles_metadata.csv  -o test/finetune/YAP1_HUMAN_Fields2012_singles_res_test.csv --epochs 2


def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments.")
    parser.add_argument("-i", "--data", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output CSV file path.")
    parser.add_argument("--checkpoint_path", type=str, default="esm2_t30_150M_UR50D", help="Model checkpoint name or full path.")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of classes (1 for regression).")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", type=str, default=("cuda:2" if torch.cuda.is_available() else "cpu"), help="Device for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay.")
    parser.add_argument("--hidden_dropout", type=float, default=0.1, help="Hidden dropout probability.")

    parser.add_argument('--lora_r', type=int, default=8, help='Rank of the low-rank decomposition.')
    parser.add_argument('--lora_alpha', type=int, default=16, help='Scaling factor for LoRA.')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='Dropout rate for LoRA.')
    parser.add_argument('--lora_modules', nargs='*', type=str, default=["q_proj", "v_proj"], help='Modules to apply LoRA. Default is q_proj, v_proj, as implemented by Microsoft.')
    return parser.parse_args()




############## Dataset ##############
class Esm2Tokenizer(Dataset):
    def __init__(self, data_file, alphabet):
        """ this class expect that data to be a dataframe with 3 columns: ID, sequence, target.
        Then the first part will convert it to a list of tuples: [(sequence_id, sequence_str, target), ...],
        as expected by batch_converter."""
        # Extract sequences and labels (targets) for batch conversion
        self.data = data_file
        sequences = [(d['ID'], d['sequence']) for _, d in self.data.iterrows()]  
        self.targets = [d['target'] for _, d in self.data.iterrows()]  
        # Convert sequences to tokens, and padding them accordingly with the longest sequence
        batch_converter = alphabet.get_batch_converter()
        self.batch_labels, self.batch_strs, self.batch_tokens = batch_converter(sequences)

    def __len__(self):
        return len(self.batch_labels)

    def __getitem__(self, idx):
        tokens = self.batch_tokens[idx]
        target = self.targets[idx]  
        target = torch.tensor(target, dtype=torch.float)
        target = target.unsqueeze(-1) # batch size x target, to match the output of the model
        return tokens, target



############## Trainer  ##############
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.model_loader = Load_from_pretrained(
            checkpoint_path=args.checkpoint_path, num_classes=args.num_classes, hidden_dropout=args.hidden_dropout,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            lora_modules=args.lora_modules)
        self.model, self.alphabet = self.model_loader.get_model_details()
        self.model.to(self.device)
        print(f"\nModel transfered to: {self.device}\n")
        self.model.print_trainable_parameters()

        print("\nLoading dataset...")
        self.train_data, self.val_data = self.load_dataset(self.args.data)

        self.train_loader = self.DataLoaders(self.train_data, self.alphabet, args.batch_size, shuffle=True)
        self.val_loader = self.DataLoaders(self.val_data, self.alphabet, args.batch_size, shuffle=False)

        print(f"Train Samples: {len(self.train_data)}, Validation Samples: {len(self.val_data)}\n")

    
    def load_dataset(self, data_path):
        data = pd.read_csv(data_path)
        train, val = train_test_split(data, test_size=0.2, random_state=42)
        return train.reset_index(drop=True), val.reset_index(drop=True)

    @staticmethod
    def DataLoaders(data, alphabet, batch_size, shuffle=False):
        dataset = Esm2Tokenizer(data, alphabet)
        return DataLoader(dataset, batch_size=batch_size, num_workers=os.cpu_count() // 2, shuffle=shuffle)


    #################### Training loop ####################
    def train_loop(self, optimizer, loss_function):
        train_predictions = []
        train_targets = []
        train_losses = []
        self.model.train()
        for tokens, targets in tqdm(self.train_loader, desc='Training: ', leave=False):
            tokens, targets = tokens.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            out = self.model(tokens)
            prediction = out['logits']
            loss = loss_function(prediction, targets)
            loss.backward() 
            optimizer.step()
            train_losses.append(loss.item())
            train_predictions.extend(prediction.detach().cpu())
            train_targets.extend(targets.cpu())

        # compute train metrics
        train_loss = np.mean(train_losses)
        train_r2 = metrics.r2_score(train_targets, train_predictions)
        train_Rho, _ = spearmanr(train_targets, train_predictions)
        return train_loss, train_r2, train_Rho


    ################## Validation loop ##################
    def eval_loop(self, loss_function):
        self.model.eval()
        val_predictions = []
        val_targets = []
        val_losses = []
        with torch.no_grad():
            for tokens, targets in tqdm(self.val_loader, desc='Validation: ', leave=False):
                tokens, targets = tokens.to(self.device), targets.to(self.device)
                out = self.model(tokens)
                prediction = out['logits']
                loss = loss_function(prediction, targets)
                val_losses.append(loss.item())
                val_predictions.extend(prediction.detach().cpu())
                val_targets.extend(targets.cpu())
        
        # compute val metrics
        val_loss = np.mean(val_losses)
        val_r2 = metrics.r2_score(val_targets, val_predictions)
        val_Rho, _ = spearmanr(val_targets, val_predictions)
        return val_loss, val_r2, val_Rho


    ################## Main training loop ##################
    def main(self):

        loss_function = nn.MSELoss()
        optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        metrics_all = {'Epoch':[], 'train_loss': [], 'train_r2': [], 'train_rho': [], 'val_loss': [], 'val_r2': [], 'val_rho': []}
        
        for epoch in range(self.args.epochs):
            train_loss, train_r2, train_rho = self.train_loop(optimizer, loss_function)
            val_loss, val_r2, val_rho = self.eval_loop(loss_function)

            # Printing and saving metrics
            metrics_all["Epoch"].append(epoch + 1)
            metrics_all["train_loss"].append(train_loss)
            metrics_all["train_r2"].append(train_r2)
            metrics_all["train_rho"].append(train_rho)
            metrics_all["val_loss"].append(val_loss)
            metrics_all["val_r2"].append(val_r2)
            metrics_all["val_rho"].append(val_rho)
            
            print(f"Epoch: {epoch+1}/{self.args.epochs}")
            print(f"|------------------------------------------|")
            print(f"| Metric |    Training    |   Validation   |")
            print(f"|--------|----------------|----------------|")
            print(f"| Loss   | {metrics_all['train_loss'][epoch]:^14.3f} | {metrics_all['val_loss'][epoch]:^14.3f} |")
            print(f"|------------------------------------------|")
            print(f"| R²     | {metrics_all['train_r2'][epoch]:^14.3f} | {metrics_all['val_r2'][epoch]:^14.3f} |")
            print(f"|------------------------------------------|")
            print(f"| Rho    | {metrics_all['train_rho'][epoch]:^14.3f} | {metrics_all['val_rho'][epoch]:^14.3f} |")
            print(f"|------------------------------------------|\n")


        # Save metrics
        output_path = Path(self.args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metrics_all).to_csv(output_path, index=False)
        print("Training complete.")



if __name__ == '__main__':
    args = parse_args()
    
    print('Training arguments:')
    print(f"Batch size: {args.batch_size}, Number of classes: {args.num_classes}, Number of epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}, Weight decay: {args.weight_decay}, Hidden layer dropout: {args.hidden_dropout}")
    print()
    
    # Initialize and run the trainer
    trainer = Trainer(args)
    trainer.main()