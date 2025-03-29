import esm
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType


# Load the pre-trained ESM-2 model
class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes, hidden_dropout):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)  # Normalize hidden states
        self.out_proj = nn.Linear(embed_dim, num_classes) 
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, features):
        x = features[:, 0, :]  # CLS token
        x = self.dense(x)
        x = self.dropout(x) 
        x = self.layer_norm(x)  # Helps stabilize training
        logits = self.out_proj(x)
        return logits



class Load_from_pretrained:
    def __init__(self, checkpoint_path, num_classes, hidden_dropout, lora_r, lora_alpha, lora_dropout, lora_modules):
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.hidden_dropout = hidden_dropout
        self.model, self.alphabet = None, None
        self.model_dimension = 0
        self._load_model()

        # LoRA parameters
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_modules = lora_modules
        self.peft_config = LoraConfig(
            inference_mode=False,
            bias="none",
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_modules)

        self.model = self._add_lora_layers()

    def _load_model(self):
        supported_models = [
            'esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D', 
            'esm2_t33_650M_UR50D', 'esm2_t36_3B_UR50D', 'esm2_t48_15B_UR50D'
            ]

        if self.checkpoint_path.endswith('.pt'):
            model_name = self.checkpoint_path.split('/')[-1].split('.')[0]
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
            else:
                print(f"Loading model {model_name} from full path and preparing for fine-tuning...")
                self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.checkpoint_path)
        
        elif self.checkpoint_path in supported_models:
            print(f"Loading model {self.checkpoint_path} and preparing for fine-tuning...")
            if self.checkpoint_path == 'esm2_t6_8M_UR50D':
                self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            elif self.checkpoint_path == 'esm2_t12_35M_UR50D':
                self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            elif self.checkpoint_path == 'esm2_t30_150M_UR50D':
                self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            elif self.checkpoint_path == 'esm2_t33_650M_UR50D':
                self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            elif self.checkpoint_path == 'esm2_t36_3B_UR50D' or self.checkpoint_path == 'esm2_t48_15B_UR50D':
                print('Those models are too big to be downloaded into the home directory.')
                print('Please download them into the working repository and pass the full checkpoints path.')
        
        elif self.checkpoint_path not in supported_models:
            raise ValueError(f"Model {model_name} not supported. Supported models are: {supported_models}")

    
        # Change the classification head to match the number of classes, regression or classification
        self.model_dimension = self.model.embed_dim
        self.model.lm_head = ClassificationHead(self.model_dimension, self.num_classes, self.hidden_dropout)


    def _add_lora_layers(self):
        # Add LoRA layer to the model
        print(f"Lora configuration: {self.peft_config}")
        self.model = get_peft_model(self.model, self.peft_config)
        return self.model
    
    def get_model_details(self):
        return self.model, self.alphabet


