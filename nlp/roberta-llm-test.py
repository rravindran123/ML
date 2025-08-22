import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from transformers import get_scheduler
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score


# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)  # Adjust num_labels accordingly

# Load a dataset (example using the IMDB dataset)
dynasent_r1 = load_dataset("dynabench/dynasent", 'dynabench.dynasent.r1.all', trust_remote_code=True)
train_texts = dynasent_r1['train']['sentence'][:2000]  # Using a subset for demonstration
train_labels = dynasent_r1['train']['gold_label'][:2000]

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

    # You can add validation logic here and print validation metrics

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-roberta')

# Note: Always include evaluation and consider using early stopping based on validation performance.