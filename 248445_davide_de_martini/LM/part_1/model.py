import torch.nn as nn

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.3,            # define layers
                 emb_dropout=0.3, n_layers=1):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)                 # Embedding layer -> idea is to use as dictionary, rows are the word embeddings
        self.dropout1 = nn.Dropout(p=emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)                   # LSTM layer
        self.pad_token = pad_index
        self.dropout2 = nn.Dropout(p=out_dropout)
        self.output = nn.Linear(hidden_size, output_size)
        
    # def forward(self, input_sequence):                                                              # define how the layers are connected
    #     emb = self.embedding(input_sequence)
    #     lstm_out, _  = self.lstm(emb)
    #     output = self.output(lstm_out).permute(0,2,1)
    #     return output 
    
    def forward(self, input_sequence):                                                              # define how the layers are connected
        emb = self.embedding(input_sequence)
        drop1 = self.dropout1(emb)
        lstm_out, _  = self.lstm(drop1)
        drop2 = self.dropout2(lstm_out)
        output = self.output(drop2).permute(0,2,1)
        return output 