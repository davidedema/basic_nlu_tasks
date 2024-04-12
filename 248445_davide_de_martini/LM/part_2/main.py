# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
import torch.optim as optim
from tqdm import tqdm
import copy
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from model import LM_LSTM

NON_MONOTONE_INTERVAL = 5

if __name__ == "__main__":
    
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")
    # divide the datased
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    # create the loaders
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

        
    hid_size = 350
    emb_size = 350

    # With SGD try with an higher learning rate (> 1 for instance)
    lr = 1.1 # This is definitely not good for SGD
    clip = 5 # Clip the gradient

    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_loss = math.inf
    best_val_loss = []
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            
            if 't0' in optimizer.param_groups[0]:       # ASGD triggered
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]['ax'].clone()
                
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)        # evaluate the model
                
                for prm in model.parameters():
                    prm.data = tmp[prm].clone()
                
            else:                                       # ASGD not triggered
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)        # evaluate the model
                
                if loss_dev < best_loss:
                    best_loss = loss_dev
                
                if 't0' not in optimizer.param_groups[0] and (len(best_val_loss) > NON_MONOTONE_INTERVAL and loss_dev > min(best_val_loss[:-NON_MONOTONE_INTERVAL])):
                    print("Triggered, switching to ASGD")
                    optimizer = optim.ASGD(model.parameters(), lr=lr, t0=0)
            
                best_val_loss.append(loss_dev)
                
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)

    folder_name = create_report_folder()
    
    generate_plots(sampled_epochs, losses_train, losses_dev, os.path.join(folder_name,"plot.png"))
    torch.save(model.state_dict(), os.path.join(folder_name, "weights.pt"))
    generate_report(sampled_epochs[-1], n_epochs, lr, hid_size, emb_size, str(type(model)), str(type(optimizer)),final_ppl, os.path.join(folder_name,"report.txt"))

    
