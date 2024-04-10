import torch.nn as nn
import torch

class VariationalDropout(nn.Module):
    def __init__(self, dropout=0.5):
        super(VariationalDropout, self).__init__()
        self.dropout = dropout
    
    def forward(self, x):
        # during evaluation we do not drop anything
        if not self.training:
            return x
        
        is_packed = isinstance(x, torch.nn.utils.rnn.PackedSequence)
        
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)
            
        mask = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        
        x = x.masked_fill(mask == 0, 0) / (1 - self.dropout)
        
        if is_packed:
            return torch.nn.utils.rnn.PackedSequence(x, batch_sizes)
        else:
            return x
        
        

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,            # define layers
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)                 # Embedding layer -> idea is to use as dictionary, rows are the word embeddings
        self.dropout1 = VariationalDropout(dropout=emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)                   # LSTM layer
        self.pad_token = pad_index
        self.dropout2 = VariationalDropout(dropout=out_dropout)
        self.output = nn.Linear(hidden_size, output_size)
        
        if emb_size == hidden_size:                                         # tie the weights of the embedding and output layers    
            self.output.weight = self.embedding.weight
    
    def forward(self, input_sequence):                                                              # define how the layers are connected
        emb = self.embedding(input_sequence)
        drop1 = self.dropout1(emb)
        lstm_out, _  = self.lstm(drop1)
        drop2 = self.dropout2(lstm_out)
        output = self.output(drop2).permute(0,2,1)
        return output 