import math
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import os

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad() 
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() 
        
    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    with torch.no_grad(): 
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

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

# Plot the training and validation loss
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

# Plot the validation perplexity
def generate_ppl_plot(epochs, perplexity_list, name):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, perplexity_list, label='Validation perplexity', marker='o')  
    plt.title('Validation Perplexity')  
    plt.xlabel('Epochs')  
    plt.ylabel('Perplexity')  
    plt.legend()  
    plt.grid(True)  
    plt.tight_layout()
    plt.savefig(name)

# Generate the report
def generate_report(epochs, number_epochs, lr, hidden_size, emb_size, model, optimizer, final_ppl, name):
    file = open(name, "w")
    file.write(f'epochs used: {epochs} \n')
    file.write(f'number epochs: {number_epochs} \n')
    file.write(f'lr: {lr} \n')
    file.write(f'hidden_size: {hidden_size} \n')
    file.write(f'embedding_size: {emb_size} \n')
    file.write(f'model: {model} \n')
    file.write(f'optimizer: {optimizer} \n')
    file.write(f'final_ppl: {final_ppl} \n')
    file.close()

# Create a new report folder with the next index
def create_report_folder():
    base_path = "/home/disi/nlu_exam/248445_davide_de_martini/LM/part_1/reports/test"
    last_index = get_last_index(os.path.dirname(base_path), os.path.basename(base_path))
    foldername = f"{base_path}{last_index + 1:02d}"
    os.mkdir(foldername)
    return foldername