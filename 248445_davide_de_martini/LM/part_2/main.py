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
import torch.optim.lr_scheduler as lr_scheduler
from model import LM_LSTM

NON_MONOTONE_INTERVAL = 5
DATASET_PATH = '/home/disi/nlu_exam/248445_davide_de_martini/LM/part_2'

if __name__ == "__main__":
    
    
    train_raw = read_file(os.path.join(DATASET_PATH, "dataset/PennTreeBank/ptb.train.txt"))
    dev_raw = read_file(os.path.join(DATASET_PATH, "dataset/PennTreeBank/ptb.valid.txt"))
    test_raw = read_file(os.path.join(DATASET_PATH, "dataset/PennTreeBank/ptb.test.txt"))
    # divide the datased
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    # create the loaders
    train_loader = DataLoader(train_dataset, batch_size=10, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

        
    hid_size = 800
    emb_size = 800 

    # With SGD try with an higher learning rate (> 1 for instance)
    lr = 5 # This is definitely not good for SGD
    clip = 5 # Clip the gradient

    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    perplexity_list = []
    sampled_epochs = []
    best_ppl = math.inf
    best_loss = math.inf
    best_val_loss = []
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    #If the PPL is too high try to change the learning rate
    try:
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
                        optimizer = optim.ASGD(model.parameters(), lr=lr, t0=0,  lambd=0.)
                
                    best_val_loss.append(loss_dev)
                    
                losses_dev.append(np.asarray(loss_dev).mean())
                perplexity_list.append(ppl_dev)
                pbar.set_description("PPL: %f" % ppl_dev)
                if  ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = 3
                else: 
                    patience -= 1
                    
                if patience <= 0:
                    break
                
            scheduler.step()
    except KeyboardInterrupt:
        pass
    
    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)

    folder_name = create_report_folder()
    
    generate_plots(sampled_epochs, losses_train, losses_dev, os.path.join(folder_name,"plot_loss.png"))
    generate_ppl_plot(sampled_epochs, perplexity_list, os.path.join(folder_name,"plot_ppl.png"))
    torch.save(model.state_dict(), os.path.join(folder_name, "weights.pt"))
    generate_report(sampled_epochs[-1], n_epochs, lr, hid_size, emb_size, str(type(model)), str(type(optimizer)),final_ppl, os.path.join(folder_name,"report.txt"))

    
