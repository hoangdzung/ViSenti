import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

from transformers import *
import numpy as np
import argparse
from tqdm import tqdm 
import random
import transformers
import pandas as pd
from collections import Counter
from data import getdata

new_version = False
if transformers.__version__ >= '2.2.2':
    new_version = True

if new_version:
    from transformers import get_linear_schedule_with_warmup
else:
    from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup


MAX_LEN = 512

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--maxlen', type=int, default=512)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=2e-5)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

MAX_LEN = args.maxlen

def get_dataloader(sentences, labels, tokenizer, batch_size):
    input_ids = []
    filtered_labels = []
    for i, sent in tqdm(enumerate(sentences)):
        encoded_sent = tokenizer.encode(sent,add_special_tokens = True)
        if len(encoded_sent) <= MAX_LEN:
            input_ids.append(encoded_sent)
            filtered_labels.append(labels[i])
    print(Counter(filtered_labels))
    inputs = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                                    value=0, truncating="post", padding="post")                                                     
    masks = []

    for sent in inputs:
        att_mask = [int(token_id > 0) for token_id in sent]
        masks.append(att_mask)

    inputs = torch.tensor(inputs)
    labels = torch.tensor(filtered_labels)
    masks = torch.tensor(masks)

    data = TensorDataset(inputs, masks, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def get_baseline_acc(model, validation_dataloader, device):
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        
        logits = outputs[1]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    eval_accuracy = eval_accuracy/nb_eval_steps
    return eval_accuracy

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=3)
model.to(device)
(X_train, Y_train), (X_test, Y_test) = getdata()
train_dataloader = get_dataloader(X_train, Y_train, tokenizer, args.batch_size)
test_dataloader = get_dataloader(X_test, Y_test, tokenizer, args.batch_size)

optimizer = AdamW(model.parameters(),lr = args.lr)
total_steps = len(train_dataloader) * args.epochs
if new_version:
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = int(0.1*total_steps),
                                            #warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
                                            #t_total = total_steps)
else:
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                        # num_warmup_steps = 0,
                                        warmup_steps = int(0.1*total_steps), # Default value in run_glue.py
                                        # num_training_steps = total_steps)
                                        t_total = total_steps)

for epoch_i in range(args.epochs):
    total_loss = 0
    model.train()
        
    # For each batch of training data...
    for step, batch in tqdm(enumerate(train_dataloader)):
        model.train()
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
                
        outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        
        loss = outputs[0]
        loss.backward()

        total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        model.zero_grad()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            
    

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    test_accuracy = get_baseline_acc(model, test_dataloader, device)
    print(" Test acc {}".format( test_accuracy))

print("")
print("Training complete!")

model = model.to('cpu')
torch.save(model, 'bert.pt')
