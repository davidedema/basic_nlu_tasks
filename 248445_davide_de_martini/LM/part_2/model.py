import torch.nn as nn
import torch

class VariationalDropout(nn.Module):
    def __init__(self, dropout=0.5):
        super(VariationalDropout, self).__init__()
        self.dropout = dropout
    
    def forward(self, x):
        
        if not self.training:
            return x
        
        batch_size = x.size(0)
        mask = torch.bernoulli((1 - self.dropout) * torch.ones(batch_size, x.size(1))).unsqueeze(2).expand_as(x)
        
        return mask * x / (1 - self.dropout)
        
        

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