import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from pathlib import Path

from dataset import *
from transformer import transformer, build_transformer

from torch.utils.tensorboard import SummaryWriter
from config import *

from tqdm import tqdm
import warnings

def get_all_senterences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang): # ds = dataset, lang = language
    """
    Get or build a tokenizer based on the provided configuration.
    
    Args:
        config: Configuration object containing tokenizer settings.
        ds: Dataset to train the tokenizer on.
        lang: Language for which the tokenizer is built.
    
    Returns:
        Tokenizer object.
    """
    print(f"Building tokenizer for {lang}...")
    print(f"Dataset size: {len(ds)} sentences")
    # print the columns of the dataset and few examples
    print(f"Dataset columns: {ds.column_names}")
    print(f"Example sentences in {lang}:")
    for i in range(5):
        print(f"{i+1}: {ds[i]['translation'][lang]}")

    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    print(f"Tokenizer path: {tokenizer_path}")
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer= WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SOS]", "[EOS]","[PAD]", "[MASK]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_senterences(ds,lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_data_set(config):
    """
    Load the dataset based on the configuration.
    
    Args:
        config: Configuration object containing dataset settings.
    
    Returns:
        Dataset object.
    """
    ds_raw = load_dataset('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # print(f"Dataset loaded with {len(ds_raw)} samples.")
    # print(f"Dataset columns: {ds_raw.column_names}")
    # print(f"Example translation from {config['lang_src']} to {config['lang_tgt']}:")
    # for i in range(5):
    #     print(f"{i+1}: {ds_raw[i]['translation'][config['lang_src']]} -> {ds_raw[i]['translation'][config['lang_tgt']]}")
    
    #Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    #split the dataset
    train_ds_size = int(0.9 * len(ds_raw))  
    val_ds_size = len(ds_raw) - train_ds_size

    # train_ds_raw, val_ds_raw = ds_raw.train_test_split(test_size=val_ds_size, seed=42).values()
    # train_ds = train_ds_raw.map(lambda x: {'src': tokenizer_src.encode(x['translation'][config['lang_src']]).ids,
    #                                         'tgt': tokenizer_tgt.encode(x['translation'][config['lang_tgt']]).ids})

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds =bilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], 
                               config['seqlen'])
    validation_ds = bilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], 
                                    config['seqlen'])
    

    max_len_src = 0
    max_len_tgt =0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    print(f"Max length source: {max_len_src}, Max length target: {max_len_tgt}")

    #create the dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    validation_dataloader = DataLoader(validation_ds, batch_size=1, shuffle=False)

    return train_dataloader, validation_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    
    model = build_transformer(
        src_vocab_size=vocab_src_len,
        tgt_vocab_size=vocab_tgt_len,
        src_seq_len=config['seqlen'],
        tgt_seq_len=config['seqlen'],
        d_model=config['d_model']
    )

    return model

# check for the device
def get_device():
    """
    Get the device to be used for training.
    
    Returns:
        torch.device: The device to be used (CPU or GPU).
    """
    # check for GPU, or mac's accelerator
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def train_model(config):
    #define the device

    device = get_device()
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, validation_dataloader, tokenizer_src, tokenizer_tgt = get_data_set(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Tensor board
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload'] is not None:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Loading model from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch= state['epoch'] +1
        optimizer.load_state_dict(state['optimizer'])
        model.load_state_dict(state['model'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
          
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", unit="batch")

        for batch in batch_iterator:

            model.train()
            encoder_input = batch['encoder_input'].to(device) # (B, seqlen  )
            decoder_input = batch['decoder_input'].to(device) 
            labels = batch['label'].to(device)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1,1 , seqlen)
            decoder_mask = batch['decoder_mask'].to(device) # (B,1, seqlen, seqlen)

            # run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seqlen, d_model)
            # decode(self, tgt, encoder_output, src_mask, tgt_mask)
            decoder_output = model.decode(encoder_output=encoder_output, src_mask=encoder_mask, 
                                          tgt=decoder_input, tgt_mask=decoder_mask) # (B, seqlen, d_model)
            projection_output = model.projection(decoder_output) # (B, seqlen, vocab_tgt_len)

            label = batch['label'].to(device) # (B, seqlen)

            #(B, seqlen, tgt_vocab_len) --> (B * seqlen, tgt_vocab_len)
            loss = loss_fn(projection_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            #log the loss to tensorboard
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.flush()

            #Backpropagate the loss
            loss.backward()

            #update the weights
            optimizer.step()
            optimizer.zero_grad()

            
            
            #save the model
            global_step += 1


        #run validation every epoch
        run_validation(model, validation_dataloader, tokenizer_src, tokenizer_tgt, config['seqlen'], device,
                        lambda msg: batch_iterator.write(msg), global_step, writer)
        # Save the model after each epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

#greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Greedily decode the output from the model.
    
    Args:
        model: The trained model.
        source: The source input tensor.
        source_mask: The mask for the source input.
        tokenizer_tgt: The target tokenizer.
        max_len: Maximum length of the output sequence.
        device: Device to run the decoding on.
    
    Returns:
        List of decoded tokens.
    """
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    pad_idx = tokenizer_tgt.token_to_id("[PAD]")

    #precompute the encoder output and reuse it for every token
    encoder_output = model.encode(source, source_mask)  # (B, seqlen, d_model)
    
    # Initialize the decoder input with SOS token
    decoder_input = torch.empty(1,1).fill_(sos_idx).long().to(device)  # (1, 1)

    while True:
        if decoder_input.size(1) > max_len:
            break
        # Create the mask for the decoder input
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)  # (1, 1, seqlen, seqlen)

        #Calculate the output of the decoder
        out = model.decode(decoder_input, encoder_output, source_mask,  decoder_mask)  # (1, seqlen, d_model)

        #get the next token probabilities
        prob = model.projection(out[:, -1])  # (1, seqlen, vocab_tgt_len)

        _, next_token = torch.max(prob, dim=-1)  # (1, seqlen)

        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)  # (1, seqlen + 1)

        if next_token == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device,
                   print_msg, global_step, writer, num_examples=5):
    """
    Run validation on the model and log results.
    Args:
        model: The model to validate.
        validation_ds: The validation dataset.
        tokenizer_src: Source tokenizer.
        tokenizer_tgt: Target tokenizer.
        max_len: Maximum length of the sequences.
        device: Device to run the validation on.
        print_msg: Function to print messages.
        global_step: Global step for logging.
        writer: TensorBoard writer for logging.
        num_examples: Number of examples to validate.
    """
    model.eval()
    count =0

    # source_texts = []
    # excepted =[]
    # predicted = []

    #size of the control window
    console_size = 80
    with torch.no_grad():
        for batch in validation_ds:
            count +=1
            encoder_input = batch['encoder_input'].to(device)  # (B, seqlen)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seqlen)
            
            assert encoder_input.size(0) == 1, "Validation batch size should be 1 for inference."

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            print(type(model_out_text))
            print(model_out_text)
            print(type(target_text))
            print(target_text)
            #print model_out_text type

        
            # source_texts.append(source_text)
            # excepted.append(target_text)
            # predicted.append(model_out_text)

            #print it to the console
            print_msg('-' * console_size)
            print_msg(f'Source: {source_text}')
            print_msg(f'Expected: {target_text}')
            print_msg(f'Predicted: {model_out_text}')
            print_msg('-' * console_size)

            model_out_text_words = [words for words in model_out_text.split()]
            target_text_words = [ words for words in target_text.split()] 

            if count >= num_examples:
                break
            
    if writer:
        #Torchmetrics, CharErrrorRate, Bleu, WordErrorRate, etc. can be used for more advanced metrics
        from torchmetrics.text.bleu import BLEUScore
        bleu = BLEUScore()
        bleu_score = bleu(model_out_text_words, target_text_words)
        print_msg(f'BLEU Score: {bleu_score.item():.4f}')
        writer.add_scalar('BLEU/validation', bleu_score.item(), global_step)
        writer.flush()

    return




if __name__ == "__main__":
    warnings.filterwarnings("ignore") # Ignore warnings for cleaner output
    config = get_config()
    train_model(config)
    print("Training complete.")

    




