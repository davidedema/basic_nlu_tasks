import torch.nn as nn
from transformers import BertModel
import os

device = 'cuda:0' 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

class ModelBert(nn.Module):

    def __init__(self, config, out_aspect, ignore_list, n_layer=1):
        super(ModelBert, self).__init__()
        
        self.num_slot = out_aspect
        self.ignore_list = ignore_list
        
        self.bert = BertModel(config)
        self.aspect_out = nn.Linear(config.hidden_size, out_aspect)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, attention_mask, input_ids, token_type_ids):
        
        bert_out = self.bert(attention_mask=attention_mask, input_ids=input_ids, token_type_ids=token_type_ids)
        last_hidden_states = bert_out.last_hidden_state
        
        drop_aspect = self.dropout(last_hidden_states)
        
        aspect_out = self.slot_out(drop_aspect)
    
        return aspect_out
        
        