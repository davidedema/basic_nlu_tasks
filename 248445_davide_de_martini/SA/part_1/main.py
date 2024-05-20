
from functions import *
from utils import *
from model import *
from tqdm import tqdm
import os
import copy
from sklearn.model_selection import train_test_split
import numpy as np
import torch.optim as optim
from transformers import BertConfig
from torch.utils.data import DataLoader

DATASET_PATH = '/home/disi/nlu_exam/248445_davide_de_martini/SA/part_1'

if __name__ == "__main__":
    
    # prepare the datasets
    tmp_train_raw = load_data(os.path.join(DATASET_PATH,'dataset','laptop_train.txt'))
    test_raw = load_data(os.path.join(DATASET_PATH,'dataset','laptop_test.txt'))
    
    # create a validation set
    train_raw, dev_raw = train_test_split(tmp_train_raw, test_size=0.1, random_state=42, shuffle=True)
    
    # create the datasets
    train_dataset = SemEvalLaptop(train_raw)
    test_dataset = SemEvalLaptop(test_raw)
    dev_dataset = SemEvalLaptop(dev_raw)
    
    # create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn,  shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=collate_fn)
    
    # get the weights for the dataset
    weights = get_weights(train_dataset)
    # multiply the weights by 3 
    weights = [x*10 for x in weights]

    conf = BertConfig.from_pretrained('bert-base-uncased')
    lr = 2e-5
    weight_decay = 0.01
    clip = 5 
    out_aspect = 3
    n_epochs = 200
    
    model = ModelBert(conf, out_aspect).to(DEVICE)
    model.apply(init_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_aspect = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0 
    pbar = tqdm(range(1,n_epochs))
    
    for x in pbar:
        loss = train_loop(train_loader, optimizer, criterion_aspect, model, clip=clip)
        # pbar.set_description(f'Loss: {np.asarray(loss).mean():.2f}')
        pbar.set_description(f'F1: {best_f1:.2f}')
        # validate every 5 epochs
        if x % 5 == 0: 
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, loss_dev = eval_loop(dev_loader, criterion_aspect, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev[2]
            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model).to(DEVICE)
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping 
                break 

    results_test, _ = eval_loop(test_loader, criterion_aspect, model)
    
    # print the results
    print('Aspect Precision', results_test[0])
    print('Aspect Recall', results_test[1])
    print('Aspect F1', results_test[2])
    
    # save the model and the results
    folder_name = create_report_folder()
    generate_plots(sampled_epochs, losses_train, losses_dev, os.path.join(folder_name,"plot.png"))
    torch.save(best_model.state_dict(), os.path.join(folder_name, "weights.pt"))
    generate_report(sampled_epochs[-1], n_epochs, lr, conf.hidden_size, str(type(model)), str(type(optimizer)), results_test[0], results_test[1], results_test[2], os.path.join(folder_name,"report.txt"))
    
    
    