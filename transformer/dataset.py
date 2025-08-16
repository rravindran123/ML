import torch
import torch.nn as nn

from torch.utils.data import Dataset

class bilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seqlen)->None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seqlen = seqlen

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens= self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens= self.seqlen - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seqlen - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Input sequence is too long for the specified sequence length.")
        

        #Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()] * enc_num_padding_tokens, dtype=torch.int64)

            ]

        )

        decoder_input= torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        #Add EOS to the target text and pad the remaining sentence
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seqlen, f"Encoder input length {encoder_input.size(0)} does not match sequence length {self.seqlen}."
        assert decoder_input.size(0) == self.seqlen, f"Decoder input length {decoder_input.size(0)} does not match sequence length {self.seqlen}."
        assert label.size(0) == self.seqlen, f"Label length {label.size(0)} does not match sequence length {self.seqlen}."

        return {
            'encoder_input': encoder_input, #seq_length,  
            'decoder_input': decoder_input, #seq_length,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #1x1xseq_length,
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() &
                causal_mask(decoder_input.size(0)), #1x1xseq_length,
            'label': label, #seq_length,
            'src_text': src_text,
            'tgt_text': tgt_text
        }

def causal_mask(size):
    """
    Create a causal mask for the decoder input.
    
    Args:
        size: The size of the mask (sequence length).
        
    Returns:
        A causal mask tensor of shape (1, 1, size, size).
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.int)
    return mask==0  
