import torch.nn as nn

# class for variational dropout
class VariationalDropout(nn.Module):
    def __init__(self, dropout=0.5):
        super(VariationalDropout, self).__init__()
        self.dropout = dropout
    
    def forward(self, x):
        # evaluation time, dropout is not applied
        if not self.training:
            return x
        
        batch_size = x.size(0)
        # create the mask
        mask = x.data.new(batch_size, 1, x.size(2)).bernoulli_(1 - self.dropout)
        mask = mask/(1-self.dropout)
        mask = mask.expand_as(x)
        # apply the mask
        return x * mask

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.5,            
                 emb_dropout=0.5, n_layers=1):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)            
        self.dropout1 = VariationalDropout(dropout=emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)                   
        self.pad_token = pad_index
        self.dropout2 = VariationalDropout(dropout=out_dropout)
        self.output = nn.Linear(hidden_size, output_size)
        
        # tie the weights
        if emb_size == hidden_size:                                           
            self.output.weight = self.embedding.weight
    
    def forward(self, input_sequence):                                                              
        emb = self.embedding(input_sequence)
        drop1 = self.dropout1(emb)
        lstm_out, _  = self.lstm(drop1)
        drop2 = self.dropout2(lstm_out)
        output = self.output(drop2).permute(0,2,1)
        return output 