import torch.nn as nn
from transformers import BertModel
import os

device = 'cuda:0' 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

class ModelBert(nn.Module):

    def __init__(self, config, out_aspect, n_layer=1):
        super(ModelBert, self).__init__()
        
        self.bert = BertModel(config)
        self.aspect_out = AspectOut(config, out_aspect)
        
    def forward(self, attention_mask, input_ids, token_type_ids):
        
        bert_out = self.bert(attention_mask=attention_mask, input_ids=input_ids, token_type_ids=token_type_ids)
        last_hidden_states = bert_out.last_hidden_state
        
        aspect_out = self.aspect_out(last_hidden_states) # [batch_size, seq_len, out_aspect]
        
        aspect_out = aspect_out.permute(0, 2, 1)  # [batch_size, out_aspect, seq_len]
    
        return aspect_out

class AspectOut(nn.Module):
    def __init__(self, config, out_aspect, dropout=0.3):
        super(AspectOut, self).__init__()
        
        self.aspect_out = nn.Linear(config.hidden_size, out_aspect)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.aspect_out(x)
        return x
        
        