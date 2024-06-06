from functions import *
from utils import *
from torch.utils.data import DataLoader
from functools import partial
import torch.optim as optim
from tqdm import tqdm
import copy
import torch.nn as nn
import math
import numpy as np
from model import LM_LSTM
import os

# Modify with the absolute path of the dataset
DATASET_PATH = '/home/davide/Desktop/248445_davide_de_martini/LM/part_1'
GEN_REPORT = False # if true, it will generate a report with the results, watch out for the folder path in the report function
TEST = True # if true, it will run the test on the test set
WEIGHTS = "weights_3.pt" # if TEST is True, it will load the weights from this file

if __name__ == "__main__":
    
    # read the dataset
    train_raw = read_file(os.path.join(DATASET_PATH, "dataset/PennTreeBank/ptb.train.txt"))
    dev_raw = read_file(os.path.join(DATASET_PATH, "dataset/PennTreeBank/ptb.valid.txt"))
    test_raw = read_file(os.path.join(DATASET_PATH, "dataset/PennTreeBank/ptb.test.txt"))
    
    # process the datasets
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    # create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=10, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

        
    hid_size = 400
    emb_size = 400

    lr = 0.0001 
    clip = 5 

    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)
    
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    
    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []    
    perplexity_list = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    
    if TEST:    
        model.load_state_dict(torch.load(os.path.join(DATASET_PATH, "bin", WEIGHTS)))
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, model)    
     
    if not TEST:
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
            if epoch % 1 == 0:  # validate every epoch
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                perplexity_list.append(ppl_dev)
                pbar.set_description("PPL: %f" % ppl_dev)
                if  ppl_dev < best_ppl: 
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = 3
                else:
                    patience -= 1
                    
                if patience <= 0: # Early stopping 
                    break 

        best_model.to(DEVICE)
    # evaluate the best model on the test set
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
            
    print('Test ppl: ', final_ppl)

    # save the model and create the report
    if GEN_REPORT:
        folder_name = create_report_folder()
        generate_plots(sampled_epochs, losses_train, losses_dev, os.path.join(folder_name,"plot.png"))
        generate_ppl_plot(sampled_epochs, perplexity_list, os.path.join(folder_name,"plot_ppl.png"))
        torch.save(best_model.state_dict(), os.path.join(folder_name, "weights.pt"))
        generate_report(sampled_epochs[-1], n_epochs, lr, hid_size, emb_size, str(type(model)), str(type(optimizer)),final_ppl, os.path.join(folder_name,"report.txt"))

    
