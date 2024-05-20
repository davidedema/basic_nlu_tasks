import torch
from evals import evaluate_ote
import torch.nn as nn
import os 
import matplotlib.pyplot as plt
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def init_weights(mat):
    '''
    Init the weights of the model
    '''
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
                    
def get_weights(dataset):
    '''
    Get the weights for the label in the dataset
    '''
    count = 0
    length = 0
    for sample in dataset:
        for aspect in sample['aspect']:
            length += 1
            if 2 == int(aspect):
                count += 1
    return [1, count/length, (length-count)/length]

def train_loop(data, optimizer, criterion_aspect, model, clip=5):
    '''
    Train the model on the training set
    '''
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() 
        aspect = model(sample['attention_mask'], sample['seq'], sample['token_type_ids'])
        loss = criterion_aspect(aspect, sample['y_aspects'])
        loss_array.append(loss.item())
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() 
    return loss_array

def eval_loop(data, criterion_aspect, model):
    '''
    Evaluate the performance of the model on the validation set
    '''
    model.eval()
    loss_array = []
    
    ref_aspect = []
    hyp_aspect = []
    ref_aspect_pad = []
    hyp_aspect_pad = []
    with torch.no_grad(): 
        for sample in data:
            aspect = model(sample['attention_mask'], sample['seq'], sample['token_type_ids'])
            loss = criterion_aspect(aspect, sample['y_aspects'])
            loss_array.append(loss.item())
            
            output_aspects = torch.argmax(aspect, dim=1)
            for id_seq, seq in enumerate(output_aspects):
                length = sample['aspect_len'].tolist()[id_seq]  
                seq_ids = sample['seq'][id_seq][:length].tolist()
                seq_ids = [int(elem) for elem in seq_ids]
                gt_aspect = sample['y_aspects'][id_seq][:length].tolist()
                # get the tokens of the sequence by converting the ids
                sequence = [tokenizer.convert_ids_to_tokens(elem) for elem in seq_ids]
                to_decode = seq[:length].tolist()
                # get the aspects from the ground truth ignoring the first and last token (pad for CLS and SEP)
                ref_aspect.append([(sequence[id_el], elem) for id_el, elem in enumerate(gt_aspect[1:-1], start=1)])
                # delete internal padding inside the aspects (pad for the words that create subtokens)
                ref_aspect_pad.append([elem for id_el, elem in enumerate(gt_aspect[1:-1], start=1) if elem != 0])
                tmp_seq = []
                # get the aspects from the model ignoring the first and last token (pad for CLS and SEP)
                for id_el, elem in enumerate(to_decode[1:-1], start=1):
                    tmp_seq.append(elem)
                hyp_aspect.append(tmp_seq)
    
    # remove tokens that are not in the reference (the one had to pad for the subtokens)
    for id_seq, seq in enumerate(ref_aspect):
        tmp_seq = []
        for id_el, elem in enumerate(seq):
            if elem[1] != 0:
                tmp_seq.append(hyp_aspect[id_seq][id_el])
        hyp_aspect_pad.append(tmp_seq)
    try:
        results = evaluate_ote(ref_aspect_pad, hyp_aspect_pad)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_aspect])
        hyp_s = set([x[1] for x in hyp_aspect])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    return results, loss_array

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
    
def generate_report(epochs, number_epochs, lr, hidden_size, model, optimizer, slot_f1, name):
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
    file.write(f'mean slot_f1: {slot_f1} \n')
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