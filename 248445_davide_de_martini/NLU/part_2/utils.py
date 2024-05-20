import torch
import torch.utils.data as data
from collections import Counter
import os
import json
from transformers import BertTokenizer

DEVICE = 'cuda:0'

PAD_TOKEN = 0

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq 
        padded_seqs = padded_seqs.detach()  
        return padded_seqs, lengths
    
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    src_utt, _ = merge(new_item['utterance'])
    
    # Build attention mask
    attention_mask = torch.LongTensor([[1 if i != PAD_TOKEN else 0 for i in seq] for seq in src_utt])
    # Build token type ids
    token_type_ids = torch.LongTensor([[0 for i in seq] for seq in src_utt])
    
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(DEVICE) 
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    token_type_ids = token_type_ids.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)
    
    new_item["utterances"] = src_utt
    new_item["attention_mask"] = attention_mask
    new_item["token_type_ids"] = token_type_ids
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item

class IntentsAndSlots (data.Dataset):
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])
            
        self.utt_ids, self.slot_ids = self.mapping_seq(self.utterances, self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, slots, mapper):
        res = []
        res_slots = []
        for seq, slot in zip(data, slots):
            tmp_seq = []
            tmp_slot = []
            for word, slot in zip(seq.split(), slot.split(' ')):
                word_tokens = self.tokenizer(word)
                # remove CLS and SEP tokens
                word_tokens['input_ids'] = word_tokens['input_ids'][1:-1]
                tmp_seq.extend(word_tokens['input_ids'])
                # extend the slot for the length of the tokenized word with the padding token
                tmp_slot.extend([mapper[slot]] + [mapper['pad']]*(len(word_tokens['input_ids'])-1))
                
            # add CLS and SEP tokens
            tmp_seq = [101] + tmp_seq + [102]
            res.append(tmp_seq)
            # add 0 and 0 to the slot corresponding to CLS and SEP tokens
            tmp_slot = [mapper['pad']] + tmp_slot + [mapper['pad']]
            res_slots.append(tmp_slot)
            
        return res, res_slots

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset