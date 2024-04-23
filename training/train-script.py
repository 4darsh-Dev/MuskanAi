# %% [code] {"id":"ywiI1z3KSRse","outputId":"42eeef5f-7072-4542-c881-456a6439afb5","jupyter":{"outputs_hidden":false}}
!pip3 install datasets transformers -q
!pip3 install wandb --upgrade -q

# %% [code] {"id":"Z9-7tD9lQaIO","jupyter":{"outputs_hidden":false}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm

from datasets import load_dataset
import random
from sklearn import metrics, model_selection, preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup

import wandb
wandb.login()

# %% [code] {"id":"Exb7P8WhQaIS","jupyter":{"outputs_hidden":false}}
def seed_everything(seed=73):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

seed_everything(1234)

# %% [code] {"id":"E4K6NpSJQaIT","jupyter":{"outputs_hidden":false}}
sweep_config = {
    'method': 'random', #grid, random, bayesian
    'metric': {
      'name': 'auc_score',
      'goal': 'maximize'   
    },
    'parameters': {

        'learning_rate': {
            'values': [5e-5, 3e-5]
        },
        'batch_size': {
            'values': [32, 64]
        },
        'epochs':{'value': 10},
        'dropout':{
            'values': [0.3, 0.4, 0.5]
        },
        'tokenizer_max_len': {'value': 40},
    }
}

sweep_defaults = {
    'learning_rate': 3e-5,
    'batch_size': 64,
    'epochs': 10,
    'dropout': 0.3,
    'tokenizer_max_len': 40
}

sweep_id = wandb.sweep(sweep_config, project='bhaavnaye')

# %% [code] {"id":"Niw-1aY6QaIU","outputId":"4f5ce478-f5e9-4444-d118-48b7a00980b8","jupyter":{"outputs_hidden":false}}
go_emotions = load_dataset("go_emotions")
data = go_emotions.data

# %% [code] {"id":"nThogcR5QaIV","jupyter":{"outputs_hidden":false}}
train, valid, test = data["train"].to_pandas(), data["validation"].to_pandas(), data["test"].to_pandas()

# %% [code] {"id":"X-Hc1rsOQaIW","outputId":"b1ab2d74-c598-48ea-a7aa-71b389b3a867","jupyter":{"outputs_hidden":false}}
print(train.shape, valid.shape, test.shape) 
train.head()

# %% [code] {"id":"J97T5BgjQaIX","jupyter":{"outputs_hidden":false}}
mapping = {0:"admiration",1:"amusement",2:"anger",3:"annoyance",4:"approval",5:"caring",6:"confusion",7:"curiosity",8:"desire",9:"disappointment",10:"disapproval",11:"disgust",12:"embarrassment",13:"excitement",14:"fear",15:"gratitude",16:"grief",17:"joy",18:"love",19:"nervousness",20:"optimism",21:"pride",22:"realization",23:"relief",24:"remorse",25:"sadness",26:"surprise",27:"neutral"}

# %% [code] {"id":"oTmlkUMIQaIY","jupyter":{"outputs_hidden":false}}
def one_hot_encoder(df):
    one_hot_encoding = []
    for i in tqdm(range(len(df))):
        temp = [0]*n_labels
        label_indices = df.iloc[i]["labels"]
        for index in label_indices:
            temp[index] = 1
        one_hot_encoding.append(temp)
    return pd.DataFrame(one_hot_encoding)

# %% [code] {"id":"bN3O37HeQaIY","outputId":"da97e909-cabf-4182-c951-3cdada722357","jupyter":{"outputs_hidden":false}}
train_ohe_labels = one_hot_encoder(train)
valid_ohe_labels = one_hot_encoder(valid)
test_ohe_labels = one_hot_encoder(test)
train_ohe_labels.shape

# %% [code] {"id":"iCeQEL0uQaIZ","jupyter":{"outputs_hidden":false}}
train = pd.concat([train, train_ohe_labels], axis=1)
valid = pd.concat([valid, valid_ohe_labels], axis=1)
test = pd.concat([test, test_ohe_labels], axis=1)
train.head()

# %% [code] {"id":"PA9lgkb0QaIa","jupyter":{"outputs_hidden":false}}
def inspect_category_wise_data(label, n=5):
    samples = train[train[label] == 1].sample(n)
    sentiment = mapping[label]
    
    print(f"{n} samples from {sentiment} sentiment: \n")
    for text in samples["text"]:
        print(text, end='\n\n')
## inspecting data
inspect_category_wise_data(4)

# %% [code] {"id":"ckGFUT8ZQaIb","jupyter":{"outputs_hidden":false}}
class GoEmotionDataset:
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels

        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        inputs = self.tokenizer.__call__(text,
                                        None,
                                        add_special_tokens=True,
                                        max_length=self.max_len,
                                        padding="max_length",
                                        truncation=True,
                                        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# %% [code] {"id":"Zxn69QX-QaIb","jupyter":{"outputs_hidden":false}}
class GoEmotionClassifier(nn.Module):
    def __init__(self, n_train_steps, n_classes, do_prob, bert_model):
        super(GoEmotionClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(do_prob)
        self.out = nn.Linear(768, n_classes)
        self.n_train_steps = n_train_steps
        self.step_scheduler_after = "batch"

    def forward(self, ids, mask):
        output_1 = self.bert(ids, attention_mask=mask)["pooler_output"]
        output_2 = self.dropout(output_1)
        output = self.out(output_2)
        return output

# %% [code] {"id":"o43mZ0TMQaIc","outputId":"5fb66934-53b0-4315-9c58-f9a0a416d647","jupyter":{"outputs_hidden":false}}
tokenizer = transformers.SqueezeBertTokenizer.from_pretrained("squeezebert/squeezebert-uncased", do_lower_case=True)

def build_dataset(tokenizer_max_len):
    train_dataset = GoEmotionDataset(train.text.tolist(), train[range(n_labels)].values.tolist(), tokenizer, tokenizer_max_len)
    valid_dataset = GoEmotionDataset(valid.text.tolist(), valid[range(n_labels)].values.tolist(), tokenizer, tokenizer_max_len)
    
    return train_dataset, valid_dataset

def build_dataloader(train_dataset, valid_dataset, batch_size):
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_data_loader, valid_data_loader

def ret_model(n_train_steps, do_prob):
  model = GoEmotionClassifier(n_train_steps, n_labels, do_prob, bert_model=bert_model)
  return model

# %% [code] {"id":"lHIN_z-XQaIc","outputId":"6ef0e472-e890-4f12-85b6-2fb51225b895","jupyter":{"outputs_hidden":false}}
sample_train_dataset, _ = build_dataset(40)
print(sample_train_dataset[0])
len(sample_train_dataset)
# loading model
bert_model = transformers.SqueezeBertModel.from_pretrained("squeezebert/squeezebert-uncased")

# %% [code] {"id":"_LZ_6Q8uQaId","jupyter":{"outputs_hidden":false}}
def ret_optimizer(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    opt = AdamW(optimizer_parameters, lr=wandb.config.learning_rate)
    return opt

def ret_scheduler(optimizer, num_train_steps):
    sch = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    return sch

def loss_fn(outputs, labels):
    if labels is None:
        return None
    return nn.BCEWithLogitsLoss()(outputs, labels.float())

def log_metrics(preds, labels):
    preds = torch.stack(preds)
    preds = preds.cpu().detach().numpy()
    labels = torch.stack(labels)
    labels = labels.cpu().detach().numpy()
    fpr_micro, tpr_micro, _ = metrics.roc_curve(labels.ravel(), preds.ravel())
    
    auc_micro = metrics.auc(fpr_micro, tpr_micro)
    return {"auc_micro": auc_micro}

# %% [code] {"id":"AyEyE6gLQaIe","jupyter":{"outputs_hidden":false}}
def train_fn(data_loader, model, optimizer, device, scheduler):
    train_loss = 0.0
    model.train()
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        mask = d["mask"]
        targets = d["labels"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask)

        loss = loss_fn(outputs, targets)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        scheduler.step()
    return train_loss
    

def eval_fn(data_loader, model, device):
    eval_loss = 0.0
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["labels"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask)
            loss = loss_fn(outputs, targets)
            eval_loss += loss.item()
            fin_targets.extend(targets)
            fin_outputs.extend(torch.sigmoid(outputs))
    return eval_loss, fin_outputs, fin_targets

# %% [code] {"id":"NCO15Qi7QaIf","jupyter":{"outputs_hidden":false}}
def trainer(config=None):
    with wandb.init(config=config):
        config = wandb.config

        train_dataset, valid_dataset = build_dataset(config.tokenizer_max_len)
        train_data_loader, valid_data_loader = build_dataloader(train_dataset, valid_dataset, config.batch_size)
        print("Length of Train Dataloader: ", len(train_data_loader))
        print("Length of Valid Dataloader: ", len(valid_data_loader))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        n_train_steps = int(len(train_dataset) / config.batch_size * 10)

        model = ret_model(n_train_steps, config.dropout)
        optimizer = ret_optimizer(model)
        scheduler = ret_scheduler(optimizer, n_train_steps)
        model.to(device)
        model = nn.DataParallel(model)
        wandb.watch(model)
        
        n_epochs = config.epochs

        best_val_loss = 100
        for epoch in tqdm(range(n_epochs)):
            train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
            eval_loss, preds, labels = eval_fn(valid_data_loader, model, device)
          
            auc_score = log_metrics(preds, labels)["auc_micro"]
            print("AUC score: ", auc_score)
            avg_train_loss, avg_val_loss = train_loss / len(train_data_loader), eval_loss / len(valid_data_loader)
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "auc_score": auc_score,
            })
            print("Average Train loss: ", avg_train_loss)
            print("Average Valid loss: ", avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "./best_model.pt")  
                print("Model saved as current val_loss is: ", best_val_loss)

# %% [code] {"id":"reQ73gRaQaIf","outputId":"9ebeb366-52b0-4485-eee1-6533a262f88d","jupyter":{"outputs_hidden":false}}
wandb.agent(sweep_id, function=trainer, count=6)
!pip freeze > requirements.txt