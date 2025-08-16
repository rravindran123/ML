import torch
import torch.nn as nn
import math

class inputEmbedding(nn.Module):
    def __init__(self, vocab_size:int, d_model:int):
        super().__init__()
        self.d_model= d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids:torch.Tensor):
        """
        input_ids: Tensor of shape (batch_size, sequence_length)
        Returns: Tensor of shape (batch_size, sequence_length, d_model)
        """
        embedded = self.embedding(input_ids) * (math.sqrt(self.d_model))    
        return embedded

class positionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model= d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len, d_model)
        #create a matrix of shape (seq_len,1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        divterm = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        #Apply the sine to even position
        pe[:, 0::2] = torch.sin(position * divterm)
        pe[:, 1::2] = torch.cos(position * divterm)

        pe= pe.unsqueeze(0) # tensor now is (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x:torch.Tensor):

        """
        x: Tensor of shape (batch_size, sequence_length, d_model)
        Returns: Tensor of shape (batch_size, sequence_length, d_model)
        """
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
        return self.dropout(x)

class layerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class feedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1, B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) #W2, B2

    def forward(self, x:torch.Tensor):
        """
        x: Tensor of shape (batch_size, sequence_length, d_model)
        Returns: Tensor of shape (batch_size, sequence_length, d_model)
        """
        x= self.linear1(x)
        x= torch.relu(x)
        x= self.dropout(x)
        x= self.linear2(x)
        return x

class multiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k= d_model // num_heads
        self.w_q= nn.Linear(d_model, d_model) #Q
        self.w_k= nn.Linear(d_model, d_model) #K
        self.w_v= nn.Linear(d_model, d_model) #V

        self.w_o = nn.Linear(d_model, d_model) #Output projection
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.size(-1)

        # query: (batch_size, num_heads, seq_len_q, d_k) ->  (batch_size, num_heads, seq_len_q, seq_len_q)
        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_scores = attention_scores.softmax(dim=-1) # batch_size, num_heads, seq_len_q, seq_len_k

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        key = self.w_k(k)
        values = self.w_v(v)
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        query = query.view(query.shape[0],query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        values = values.view(values.shape[0], values.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        # (batch_size, num_heads, seq_len, d_k)

        x, self.attention_scores = multiHeadAttention.attention(query, key, values, mask, self.dropout)

        # (batch, num_heads, seq_len, d_k) --> (batch, seq_len, num_heads, d_k) --> (batch, seq_len, d_model)
        x=  x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k) 

        return self.w_o(x) # (batch_size, seq_len, d_model)
    
class residualConnection(nn.Module):

    def __init__(self, features:int, dropout:float=0.1)-> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = layerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class encoderBlock(nn.Module):
    def __init__(self, features:int, self_attention_block:multiHeadAttention, feed_forward_block:feedForward, dropout:float=0.1):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([residualConnection(features, dropout) for _ in range(2)])  

    
    def forward(self, x, src_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x= self.residual_connections[1](x, self.feed_forward_block)
        return x

class encoder(nn.Module):

    def __init__(self, features:int , layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = layerNormalization(features)

    def forward(self, x, src_mask=None):
        for layers in self.layers:
            x = layers(x, src_mask)
        return self.norm(x)
    
class decoderBlock(nn.Module):

    def __init__(self, features:int, self_attention_block:multiHeadAttention, 
                 cross_attention_block:multiHeadAttention, 
                 feed_forward_block:feedForward, dropout:float=0.1):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([residualConnection(features, dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x= self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x= self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x= self.residual_connections[2](x, self.feed_forward_block)
        return x
    

class decoder(nn.Module):
    def __init__(self, features:int, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = layerNormalization(features)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class projection(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x:torch.Tensor):
        """
        x: Tensor of shape (batch_size, sequence_length, d_model)
        Returns: Tensor of shape (batch_size, sequence_length, vocab_size)
        """ 
        x = self.linear(x)
        #return torch.log_softmax(x)
        return x
    

class transformer(nn.Module):

    def __init__(self, _encoder:encoder, _decoder:decoder, _projection:projection, src_embed:inputEmbedding, tgt_embed:inputEmbedding,
                  src_pos: positionalEncoding, tgt_pos:positionalEncoding) -> None:
        super().__init__()
        self.encoder = _encoder
        self.decoder = _decoder
        self.projection = _projection
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection(x)
    

def build_transformer(src_vocab_size:int , tgt_vocab_size:int, src_seq_len:int, tgt_seq_len:int, 
                      d_model:int=512, N:int=6, num_heads:int=8, d_ff:int=2048, dropout:float=0.1):
    #create the embedding layers
    src_embed = inputEmbedding(src_vocab_size, d_model)
    tgt_embed = inputEmbedding(tgt_vocab_size, d_model)
    
    #create the positional encoding layers
    src_pos = positionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = positionalEncoding(d_model, tgt_seq_len, dropout)

    #create the encoder block
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block = multiHeadAttention(d_model, num_heads, dropout)
        feed_forward_block = feedForward(d_model, d_ff, dropout)
        encoder_block= encoderBlock(d_model,encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    #create the decoder block
    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block = multiHeadAttention(d_model, num_heads, dropout)
        decoder_cross_attention_block = multiHeadAttention(d_model, num_heads, dropout)
        feed_forward_block = feedForward(d_model, d_ff, dropout)
        decoder_block = decoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    #create the encoder and decoder
    _encoder = encoder(d_model, nn.ModuleList(encoder_blocks))
    _decoder = decoder(d_model, nn.ModuleList(decoder_blocks))

    #projection layer
    projection_layer = projection(d_model, tgt_vocab_size)

    transformer_model = transformer(_encoder, _decoder, projection_layer, src_embed, tgt_embed, src_pos, tgt_pos)

    #initialize the parameters
    for p in transformer_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


    return transformer_model  










