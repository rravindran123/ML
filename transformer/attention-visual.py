import torch
import torch.nn as nn
from transformer import *
from config import *
from train import *
from dataset import *
import altair as alt
import pandas as pd
import warnings, os
warnings.filterwarnings("ignore")

def get_device():
    #check for cpu, gpu and mps
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device

device = get_device()

config = get_config()
train_dataloader, val_dataloader, vocab_src, vocab_tgt = get_data_set(config)
model = get_model(config, vocab_src.get_vocab_size(), vocab_tgt.get_vocab_size()).to(device)

# Load the pretrained weights
model_filename = get_weights_file_path(config, f"19")
print(f"Loading model from {model_filename}")
if not os.path.exists(model_filename):
    raise FileNotFoundError(f"Model file {model_filename} does not exist.")
state = torch.load(model_filename)
print(f"Available keys in state dictionary: {state.keys()}")
if 'model_state_dict' not in state:
    raise KeyError(f"'model_state_dict' key not found in the state dictionary. Available keys: {state.keys()}")
model.load_state_dict(state['model_state_dict'])

def load_next_batch():
    batch = next(iter(train_dataloader))
    encoder_input = batch['encoder_input'].to(device)
    encoder_mask = batch['encoder_mask'].to(device)
    decoder_input = batch['decoder_input'].to(device)
    decoder_mask = batch['decoder_mask'].to(device)

    encoder_input_tokens = [vocab_src.lookup_token(token.item()) for token in encoder_input[0]]
    decoder_input_tokens = [vocab_tgt.lookup_token(token.item()) for token in decoder_input[0]]