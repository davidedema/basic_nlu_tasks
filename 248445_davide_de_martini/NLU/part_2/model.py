import torch.nn as nn
import torch
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os

device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0

class ModelBert(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, intent_label_lst, slot_label_lst, ignore_list, n_layer=1, pad_index=0):
        super(ModelBert, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        
        self.num_intent = len(intent_label_lst)
        self.num_slot = len(slot_label_lst)
        self.ignore_list = ignore_list
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, attention_mask, input_ids, token_type_ids, intent_labels, slot_labels):
        
        bert_out = self.bert(attention_mask=attention_mask, input_ids=input_ids, token_type_ids=token_type_ids)
        last_hidden_states = bert_out.last_hidden_state
        pooled_output = bert_out.pooler_output
        
        drop_slot = self.dropout(last_hidden_states)
        drop_intent = self.dropout(pooled_output)
        
        slot_out = self.slot_out(drop_slot)
        intent_out = self.intent_out(drop_intent)
        
        #? return intent_out, slot_out
        
        # TODO: Capire se fare loss dentro il forward o fuori (finire di vedere)
        
        
        combined_loss = 0
        
        # softmax for intent classification
        if intent_labels is not None:
            if self.num_intent > 1:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_out.view(-1, self.num_intents), intent_labels.view(-1))
            elif self.num_intent == 1: 
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_out.view(-1), intent_labels.view(-1))
            
            combined_loss += intent_loss
        
        # softmax for slot filling
        if slot_labels is not None:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_list)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_out.view(-1, self.num_slot)[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]
                slot_loss = slot_loss_fct(active_logits, active_labels)
            else:
                slot_loss = slot_loss_fct(slot_out.view(-1, self.num_slot), slot_labels.view(-1))
            
            combined_loss += slot_loss
        
        outputs = ((intent_out, slot_out),) + outputs[2:]
        outputs = (combined_loss,) + outputs
    
        return outputs
        
        