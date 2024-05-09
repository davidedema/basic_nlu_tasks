import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from collections import Counter
import os
import json
from transformers import BertTokenizer

device = 'cuda:0'

PAD_TOKEN = 0
SEP_TOKEN = 102

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

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

# put all the things for BERT

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    # Build attention mask
    attention_mask = torch.LongTensor([[1 if i != PAD_TOKEN else 0 for i in seq] for seq in src_utt])
    # Build token type ids
    token_type_ids = torch.LongTensor([[0 for i in seq] for seq in src_utt])
    y_aspect, y_lengths = merge(new_item["aspect"])
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_aspect = y_aspect.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["utterances"] = src_utt
    new_item["attention_mask"] = attention_mask
    new_item["token_type_ids"] = token_type_ids
    new_item["y_aspects"] = y_aspect
    new_item["aspect_len"] = y_lengths
    return new_item

class SemEvalLaptop(data.Dataset):
    def __init__(self, dataset, lang=None, unk='unk'):
        self.utterances = []
        self.aspects = []
        self.unk = unk
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.aspects.append(x['aspect'])
            
        self.utt_ids, self.aspects_ids = self.mapping_seq(self.utterances, self.aspects)
        
    def __len__(self):
        return len(self.utterances)
    
    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        aspect = torch.Tensor(self.aspects_ids[idx])
        sample = {'utterance': utt, 'aspect': aspect}
        return sample
    
    def mapping_seq(self, utterances, aspects):
        utt_ids = []
        aspect_ids = []
        for utt, aspect in zip(utterances, aspects):
            tmp_seq = []
            tmp_aspect = []
            for word, tag in zip(utt.split(), aspect):
                # handle the case 'word.'
                if word[-1] == '.':
                    word = word[:-1]
                    tokens = self.tokenizer(word)
                    tokens['input_ids'] = tokens['input_ids'][1:-1]
                    tmp_seq.extend(tokens['input_ids'])
                    tmp_aspect.extend([tag]*len(tokens['input_ids']))
                    dot = self.tokenizer('.')
                    tmp_seq.extend(dot['input_ids'][1:-1])
                    tmp_aspect.extend([0]*len(dot['input_ids'][1:-1]))
                else:
                    tokens = self.tokenizer(word)
                    tokens['input_ids'] = tokens['input_ids'][1:-1]
                    tmp_seq.extend(tokens['input_ids'])
                    tmp_aspect.extend([tag]*len(tokens['input_ids']))
            
            # add CLS and SEP tokens
            tmp_seq = [101] + tmp_seq + [102]            
            utt_ids.append(tmp_seq)
            # add 0 and 0 to the aspect for CLS and SEP tokens
            tmp_aspect = [0] + tmp_aspect + [0]
            aspect_ids.append(tmp_aspect)
            
        return utt_ids, aspect_ids

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        for line in f.readlines():
            x = {}
            sent, tag = line.strip().split('####')
            x['utterance'] = sent
            tag = tag.split()
            tmp = []
            for char in tag:
                if char[0] != '=':
                    _, char = char.split('=')
                else:
                    char = char[2:]
                t = char
                if t == 'O':
                    tmp.append(100)
                else:
                    tmp.append(200)
            x['aspect'] = tmp
            dataset.append(x)
    return dataset