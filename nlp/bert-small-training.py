import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_scheduler
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import os, wget
import pandas as pd
from collections import defaultdict, Counter

# Load tokenizer and model using a smaller BERT variant
#tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
#model = BertForSequenceClassification.from_pretrained('google/bert_uncased_L-2_H-128_A-2', num_labels=3)  # Adjust num_labels accordingly


def print_label_dist(dataset, labelname='gold_label', splitnames=('train', 'validation')):
    for splitname in splitnames:
        print(splitname)
        dist = sorted(Counter(dataset[splitname][labelname]).items())
        for k, v in dist:
            print(f"\t{k:>14s}: {v}")

    
tokenizer = BertTokenizer.from_pretrained('/home/ubuntu/ML/fine-tuned-bert')
model = BertForSequenceClassification.from_pretrained('/home/ubuntu/ML/fine-tuned-bert', num_labels=3)  # Adjust num_labels accordingly

def train(train_texts, train_labels):
    # Convert labels to integers if they are not already
    label_to_index = {'neutral': 0, 'positive': 1, 'negative': 2}  # Update this mapping as per your labels
    train_labels = [label_to_index[label] for label in train_labels]

    # Tokenization
    encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

    # Dataset and DataLoader
    class IMDBDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = IMDBDataset(encodings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Move model to GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Print average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Training Loss: {avg_loss:.4f}")

    # Save the fine-tuned model
    model.save_pretrained('./fine-tuned-bert')

    # Also save the tokenizer used with the model
    tokenizer.save_pretrained('./fine-tuned-bert')

'''
    # Load a dataset (example using the Dynasent dataset)
dynasent_r1 = load_dataset("dynabench/dynasent", 'dynabench.dynasent.r1.all', trust_remote_code=True)
train_texts = dynasent_r1['train']['sentence']  # Using a subset for demonstration
train_labels = dynasent_r1['train']['gold_label']
'''
'''
    # Load a dataset (example using the Dynasent dataset)
dynasent_r2 = load_dataset("dynabench/dynasent", 'dynabench.dynasent.r2.all', trust_remote_code=True)
train_texts = dynasent_r2['train']['sentence']  # Using a subset for demonstration
train_labels = dynasent_r2['train']['gold_label']
'''
sst = load_dataset("SetFit/sst5")

print(sst)

print_label_dist(sst, labelname='label_text')

def convert_sst_label(s):
    return s.split(" ")[-1]

for splitname in ('train', 'validation', 'test'):
    dist = [convert_sst_label(s) for s in sst[splitname]['label_text']]
    sst[splitname] = sst[splitname].add_column('gold_label', dist)
    sst[splitname] = sst[splitname].add_column('sentence', sst[splitname]['text'])

train_texts = sst['train']['sentence']  # Using a subset for demonstration
train_labels = sst['train']['gold_label']

train(train_texts, train_labels)

'''
if not os.path.exists(os.path.join("data", "sentiment", "cs224u-sentiment-test-unlabeled.csv")):
    os.makedirs(os.path.join('data', 'sentiment'), exist_ok=True)
    wget.download('https://web.stanford.edu/class/cs224u/data/cs224u-sentiment-test-unlabeled.csv', out='data/sentiment/')

bakeoff_df = pd.read_csv("https://web.stanford.edu/class/cs224u/data/cs224u-sentiment-test-unlabeled.csv")
print(bakeoff_df.head())

def predict_labels(text_list):
    # Your method to get predictions from the model
    # This function should return a list of predictions corresponding to the input text list
    #return model.predict(text_list)
    model.eval()
    return model(text_list)

bakeoff_df['prediction'] = predict_labels(bakeoff_df['sentence'].tolist())


bakeoff_df.to_csv("data/sentiment/cs224u-sentiment-bakeoff-entry.csv", index=False)
'''