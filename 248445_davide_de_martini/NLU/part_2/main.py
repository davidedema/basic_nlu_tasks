
from functions import *
from utils import *
from model import *
from tqdm import tqdm
import copy
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from collections import Counter
from transformers import BertConfig

# Modify with the absolute path of the dataset
DATASET_PATH = '/home/davide/Desktop/248445_davide_de_martini/NLU/part_2'

GEN_REPORT = False # if true, it will generate a report with the results, watch out for the folder path in the report function
TEST = True # if true, it will run the test on the test set
WEIGHTS = "weights.pt" # if TEST is True, it will load the weights from this file

if __name__ == "__main__":
    
    saving_object = None
    
    if TEST:
        saving_object = torch.load(os.path.join(DATASET_PATH,'bin','weights.pt'))
    
    # preprocess the data and prepare the datasets
    tmp_train_raw = load_data(os.path.join(DATASET_PATH,'dataset','train.json'))
    test_raw = load_data(os.path.join(DATASET_PATH,'dataset','test.json'))
    
    portion = 0.10

    intents = [x['intent'] for x in tmp_train_raw] 
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: 
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    
    # Create the train and dev set
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, random_state=42, shuffle=True, stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x['intent'] for x in test_raw]
    
    words = sum([x['utterance'].split() for x in train_raw], []) 
    
    corpus = train_raw + dev_raw + test_raw 
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0)
    
    if TEST:
        lang.word2id = saving_object['w2id']
        lang.slot2id = saving_object['slot2id']
        lang.intent2id = saving_object['intent2id']
    
    # create the datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    conf = BertConfig.from_pretrained('bert-base-uncased')
    
    lr = 0.0001 
    clip = 5 

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    n_epochs = 200
    ignore_list = 102
    pbar = tqdm(range(1, n_epochs))

    model = ModelBert(conf, out_slot, out_int, ignore_list).to(DEVICE)
    model.apply(init_weights) 
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() 
    
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0 
    
    if TEST:
        model.load_state_dict(saving_object['model'])
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
        
    if not TEST:
        for x in pbar:
            loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
            # validate every 5 epochs
            if x % 5 == 0: 
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                
                f1 = results_dev['total']['f']
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = 3
                else:
                    patience -= 1
                if patience <= 0: # Early stopping 
                    break 

        best_model.to(DEVICE)
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang)
    intent_acc = intent_test['accuracy'] 
    slot_f1 = results_test['total']['f']   
    
    # print the results
    print('Slot F1', slot_f1)
    print('Intent Acc', intent_acc)
    
    # save the model and the results
    if GEN_REPORT:
        folder_name = create_report_folder()
        generate_plots(sampled_epochs, losses_train, losses_dev, os.path.join(folder_name,"plot.png"))
        PATH = os.path.join("bin", "weights_1.pt")
        saving_object = {"epoch": x, 
                        "model": model.state_dict(), 
                        "optimizer": optimizer.state_dict(), 
                        "w2id": lang.word2id, 
                        "slot2id": lang.slot2id, 
                        "intent2id": lang.intent2id}
        torch.save(saving_object, PATH)
        generate_report(sampled_epochs[-1], n_epochs, lr, conf.hidden_size, str(type(model)), str(type(optimizer)), slot_f1, intent_acc, os.path.join(folder_name,"report.txt"))
    
    
    