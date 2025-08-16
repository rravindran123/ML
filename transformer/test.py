from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

# # Define the tokenizer
# tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
# tokenizer.pre_tokenizer = Whitespace()

# # Define special tokens and train the tokenizer
# special_tokens = ["[UNK]", "[CLS]", "[SOS]", "[EOS]", "[PAD]", "[MASK]"]
# trainer = WordLevelTrainer(special_tokens=special_tokens, min_frequency=1)  # Reduced min_frequency to 1

# # Example sentences
# sentences = [
#     "I love programming.",
#     "Transformers are powerful models.",
#     "Tokenization is an important step."
# ]

# # Train the tokenizer
# tokenizer.train_from_iterator(sentences, trainer=trainer)

# # Add special tokens during preprocessing
# max_length = 8
# for sentence in sentences:
#     tokenized = tokenizer.encode(f"[SOS] {sentence} [EOS]")
#     padded_tokens = tokenized.tokens + ["[PAD]"] * (max_length - len(tokenized.tokens))
#     print(padded_tokens)

import torch
#import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class wineDataset(Dataset):
    def __init__(self):
        #check the file before executing
        with open('./wine.csv', 'r') as f:
            header = f.readline().strip()
            print("Header:", header)
        xy = np.loadtxt('./wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]
        print(self.x.shape, self.y.shape)
    

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    

    def __len__(self):
        return self.n_samples
    
# if __name__ == "__main__":
ds = wineDataset()
dataloader = DataLoader(dataset=ds, batch_size=4, shuffle=True, num_workers=1)

# dataiter = iter(dataloader)
# sample = next(dataiter)
# features, labels = sample
# print(features, labels)

#training loop
num_epochs = 2
total_samples = len(ds)

n_iterations = math.ceil(total_samples / 4)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # Forward pass
        # loss = model(inputs, labels)
        # Backward pass
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        if (i + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_iterations}], Inputs: {inputs.shape}, Labels: {labels.shape}')