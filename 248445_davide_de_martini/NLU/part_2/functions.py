
import torch
import torch.nn as nn
from conll import evaluate
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt

device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0

def init_weights(mat):
    for m in mat.modules():
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
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
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
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
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
    
def generate_report(runs, epochs, number_epochs, lr, hidden_size, emb_size, model, optimizer, slot_f1, intent_acc, slot_f1_std, intent_acc_std, name):
    file = open(name, "w")
    file.write(f'runs: {runs} \n')
    file.write(f'epochs used: {epochs} \n')
    file.write(f'number epochs: {number_epochs} \n')
    file.write(f'lr: {lr} \n')
    file.write(f'hidden_size: {hidden_size} \n')
    file.write(f'embedding_size: {emb_size} \n')
    file.write(f'model: {model} \n')
    file.write(f'optimizer: {optimizer} \n')
    file.write(f'mean slot_f1: {slot_f1} variance {slot_f1_std}\n')
    file.write(f'mean intent_acc: {intent_acc} variance {intent_acc_std} \n')
    file.close()

def create_report_folder():
    base_path = "/home/davide/Desktop/nlu_exam/248445_davide_de_martini/NLU/part_2/reports/test"
    last_index = get_last_index(os.path.dirname(base_path), os.path.basename(base_path))
    foldername = f"{base_path}{last_index + 1:02d}"
    os.mkdir(foldername)
    return foldername