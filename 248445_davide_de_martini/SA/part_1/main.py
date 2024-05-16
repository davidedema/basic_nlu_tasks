
from functions import *
from utils import *
from model import *
from tqdm import tqdm
import os
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import BertConfig


device = 'cuda:0' 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
PAD_TOKEN = 0
DATASET_PATH = '/home/disi/nlu_exam/248445_davide_de_martini/SA/part_1'

if __name__ == "__main__":
    
    train_raw = load_data(os.path.join(DATASET_PATH,'dataset','overfit.txt'))
    test_raw = load_data(os.path.join(DATASET_PATH,'dataset','overfit.txt'))
    
    train_dataset = SemEvalLaptop(train_raw)
    test_dataset = SemEvalLaptop(test_raw)
    
    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn,  shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    conf = BertConfig.from_pretrained('bert-base-uncased')
    
    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient
    
    out_aspect = 3
    
    n_epochs = 200
    aspect_f1s, intent_acc = [], []
    
    model = ModelBert(conf, out_aspect).to(device)
    # model.apply(init_weights) 
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_aspect = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0 
    
    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion_aspect, model, clip=clip)
        if x % 1 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, loss_dev = eval_loop(test_loader, criterion_aspect, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev[2]
            print(f1, best_f1)
            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
                # Here you should save the model
                patience = 3
            else:
                patience -= 1
            # if patience <= 0: # Early stopping with patience
            #     break # Not nice but it keeps the code clean

    results_test, _ = eval_loop(test_loader, criterion_aspect, model)   
    aspect_f1s.append(results_test[2]) 
        
    aspect_f1s = np.asarray(aspect_f1s)
    
    print('Aspect F1', round(aspect_f1s.mean(),3), '+-', round(aspect_f1s.std(),3))
    
    folder_name = create_report_folder()
    generate_plots(sampled_epochs, losses_train, losses_dev, os.path.join(folder_name,"plot.png"))
    # saving_object = {"epoch": x, 
    #                  "model": model.state_dict(), 
    #                  "optimizer": optimizer.state_dict(), 
    #                  "w2id": w2id, 
    #                  "slot2id": slot2id, 
    #                  "intent2id": intent2id}
    # torch.save(saving_object, PATH)
    torch.save(model.state_dict(), os.path.join(folder_name, "weights.pt"))
    generate_report(sampled_epochs[-1], n_epochs, lr, conf.hidden_size, str(type(model)), str(type(optimizer)), round(aspect_f1s.mean(),3), round(intent_acc.mean(), 3), round(aspect_f1s.std(),3), round(aspect_f1s.std(), 3), os.path.join(folder_name,"report.txt"))
    
    
    