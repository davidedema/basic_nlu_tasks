
import torch
import torch.nn as nn
from conll import evaluate
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def init_weights(mat):
    '''
    Init the weights of the model
    '''
    for m in mat.modules():
        # Do not init BertModel layers, they are pretrained
        if isinstance(m, (BertModel,)):
                # print("Skipping", m)
                continue  # Skip BertModel layers
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    '''
    Train the model on the training set
    '''
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() 
        intent, slots = model(sample['attention_mask'], sample['utterances'], sample['token_type_ids'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot 
        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() 
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    '''
    Evaluate the performance of the model on the validation set
    '''
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    ref_slots_pad = []
    hyp_slots_pad = []
    with torch.no_grad(): 
        for sample in data:
            intents, slots = model(sample['attention_mask'], sample['utterances'], sample['token_type_ids'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            
            # Intent inference
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                utt_ids = [int(elem) for elem in utt_ids]
                gt_ids = sample['y_slots'][id_seq].tolist()
                # get the tokens of the utterance by converting the ids
                utterance = [tokenizer.convert_ids_to_tokens(elem) for elem in utt_ids]
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                to_decode = seq[:length].tolist()
                # get the slots from the ground truth ignoring the first and last token (pad for CLS and SEP)
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots[1:-1], start=1)])
                # delete internal padding inside the slots (pad for the words that create subtokens)
                ref_slots_pad.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots[1:-1], start=1) if elem != 'pad'])
                tmp_seq = []
                # get the slots from the model ignoring the first and last token (pad for CLS and SEP)
                for id_el, elem in enumerate(to_decode[1:-1], start=1):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    
    
    # remove tokens that are not in the reference (the one had to pad for the subtokens)
    for id_seq, seq in enumerate(ref_slots):
        tmp_seq = []
        for id_el, elem in enumerate(seq):
            if elem[1] != 'pad':
                tmp_seq.append(hyp_slots[id_seq][id_el])
        hyp_slots_pad.append(tmp_seq)
                              
    try:
        results = evaluate(ref_slots_pad, hyp_slots_pad)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def get_last_index(directory, base_name):
    '''
    Function used to get the last index of the file in the directory (for the report folder)
    '''
    # Get a list of all files in the directory
    files = os.listdir(directory)
    # Filter out only the files with the specified base name
    indices = []
    for file in files:
        if file.startswith(base_name):
            try:
                index = int(str(file[len(base_name):]))  # Extracting the numeric part
                indices.append(index)
            except ValueError:
                pass
    # Return the maximum index if files exist, otherwise return 0
    return max(indices) if indices else -1

def generate_plots(epochs, loss_train, loss_validation, name):
    '''
    Draw the plot of the loss function
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_train, label='Training Loss', marker='o')  
    plt.plot(epochs, loss_validation, label='Validation Loss', marker='s')  
    plt.title('Training and Validation Loss')  
    plt.xlabel('Epochs')  
    plt.ylabel('Loss')  
    plt.legend()  
    plt.grid(True)  
    plt.tight_layout()
    plt.savefig(name)

def generate_report(epochs, number_epochs, lr, hidden_size, model, optimizer, slot_f1, intent_acc, name):
    '''
    Generate a report with the results of the test
    '''
    file = open(name, "w")
    file.write(f'epochs used: {epochs} \n')
    file.write(f'number epochs: {number_epochs} \n')
    file.write(f'lr: {lr} \n')
    file.write(f'hidden_size: {hidden_size} \n')
    file.write(f'model: {model} \n')
    file.write(f'optimizer: {optimizer} \n')
    file.write(f'slot_f1: {slot_f1} \n')
    file.write(f'intent_acc: {intent_acc} \n')
    file.close()

def create_report_folder():
    '''
    Create folder contating all the info for the test
    '''
    base_path = "/home/disi/nlu_exam/248445_davide_de_martini/NLU/part_2/reports/test"
    last_index = get_last_index(os.path.dirname(base_path), os.path.basename(base_path))
    foldername = f"{base_path}{last_index + 1:02d}"
    os.mkdir(foldername)
    return foldername
