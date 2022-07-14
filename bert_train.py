import argparse, os, json
import pandas as pd
import torch, transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils_common import Triage

# Setting up the device for GPU usage
from torch import cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, LABEL_SIZE)

    def forward(self, input_ids, attention_mask):
        output_1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def train(epoch):
    tr_loss, n_correct = 0, 0
    nb_tr_steps, nb_tr_examples = 0, 0
    
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)
        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)
        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        if _%(len(training_loader)//10)==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples 
            print("Training Loss per 1/10 epoch: {:.4f}".format(loss_step))
            print("Training Accuracy per 1/10 epoch: {:.2f}%".format(accu_step))

        optimizer.zero_grad()
        loss.backward()
        # When using GPU
        optimizer.step()
        scheduler.step()

    print("The Total Accuracy for Epoch {:.2f}%".format((n_correct*100)/nb_tr_examples))
    epoch_loss = tr_loss/nb_tr_steps
    print("Training Loss Epoch: {:.4f}".format(epoch_loss))
    return


def valid(model, testing_loader):
    model.eval()
    tr_loss = 0; nb_tr_steps = 0; nb_tr_examples = 0
    n_correct = 0; n_wrong = 0; total = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)
            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print("Validation Loss Epoch: {:.4f}".format(epoch_loss))
    return epoch_accu


def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x]=len(encode_dict)
    return encode_dict[x]

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-p', '--pretrain_path', type=str, required=True)
    parser.add_argument('-t', '--title_column', type=str, required=True)
    parser.add_argument('-l', '--label', type=str, required=True)
    parser.add_argument('-o', '--saved_path', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    input_file = args.input
    pretrain_model = args.pretrain_path
    title_column = args.title_column
    train_label = args.label
    saved_path = args.saved_path
    
    ## read csv
    df = pd.read_csv(input_file)
    df = df[[title_column,train_label]]
    
    df = df[~df[train_label].isna()]
    df = df[~df[title_column].isna()]
    df[train_label] = df[train_label].astype(int)
    
    df.rename(columns={title_column:'TITLE'}, inplace=True)
    df.dropna(subset=['TITLE', train_label], inplace=True)
    df = df.reset_index(drop=True)

    encode_dict = {}
    df['ENCODE_CAT'] = df[train_label].astype(int).apply(lambda x: encode_cat(x))
    decode_dict = {y:x for x,y in encode_dict.items()}
    
    # Hyperparameter
    MAX_LEN = 64
    TRAIN_BATCH_SIZE = 256
    VALID_BATCH_SIZE = 128
    EPOCHS = 5
    LEARNING_RATE = 2e-05
    LABEL_SIZE = df['ENCODE_CAT'].nunique()
    
    # create dataloader
    train_size = 0.8
    train_dataset=df.sample(frac=train_size,random_state=200)
    test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    
    print("===="*20)
    print("===="*20)
    print("the model will have {} labels !".format(LABEL_SIZE))
    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))
    print("===="*20)
    print("===="*20)
    tokenizer = BertTokenizer.from_pretrained(pretrain_model)
    training_set = Triage(train_dataset, tokenizer, MAX_LEN)
    testing_set = Triage(test_dataset, tokenizer, MAX_LEN)
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 8,
                    }
    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 8,
                  }
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    
    # model and loss funciton
    model = BERTClass()
    model = nn.DataParallel(model, device_ids=[ 0,1]).to(device)
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(params = model.parameters(), lr=LEARNING_RATE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(training_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps,
                                                )


    for epoch in range(EPOCHS):
        print("Epoch {}:\n".format(epoch))
        train(epoch)
        acc = valid(model, testing_loader)
        print("Accuracy on test data = %0.2f%%" % acc)

        # Saving the files for re-use
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        output_model_file = os.path.join(saved_path, 'pytorch_model.bin')
        output_vocab_file = os.path.join(saved_path, 'vocab.txt')
        model_to_save = model
        torch.save(model_to_save.module.state_dict(), output_model_file)
        
        tokenizer.save_vocabulary(output_vocab_file)

        with open(os.path.join(saved_path, 'label_decoder.json'), 'w') as fi:
            json.dump(decode_dict, fi)
        model.module.bert.config.to_json_file(os.path.join(saved_path, 'config.json'))

    print('All files saved in {}'.format(saved_path))
    print('model training finished!')