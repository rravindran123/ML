try:
    # Sort of randomly chosen import to see whether the requirements
    # are met:
    import datasets
except ModuleNotFoundError:
  #  !git clone https://github.com/cgpotts/cs224u/
  #  !pip install -r cs224u/requirements.txt
    import sys
    sys.path.append("cs224u")
import sys
sys.path.append("cs224u")
from collections import defaultdict, Counter
from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import transformers
from transformers import AutoModel, AutoTokenizer
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
from transformers import RobertaTokenizer
import torch.nn as nn
import torch
import os
import wget
import re
import random

#import the data sets


dynasent_r1 = load_dataset("dynabench/dynasent", 'dynabench.dynasent.r1.all', trust_remote_code=True)
print(dynasent_r1)

def print_label_dist(dataset, labelname='gold_label', splitnames=('train', 'validation')):
    for splitname in splitnames:
        print(splitname)
        dist = sorted(Counter(dataset[splitname][labelname]).items())
        for k, v in dist:
            print(f"\t{k:>14s}: {v}")

print_label_dist(dynasent_r1)

train_data = dynasent_r1['train'].shuffle(seed=42).select(range(100))

validation_data = dynasent_r1['validation'].shuffle(seed=42).select(range(100))
#test_data = dynasent_r1['test'].shuffle(seed=51).select(range(1000))

dynasent_r2 = load_dataset("dynabench/dynasent", 'dynabench.dynasent.r2.all')

print_label_dist(dynasent_r2)

sst = load_dataset("SetFit/sst5")

print(sst)

print_label_dist(sst, labelname='label_text')

def convert_sst_label(s):
    return s.split(" ")[-1]

for splitname in ('train', 'validation', 'test'):
    dist = [convert_sst_label(s) for s in sst[splitname]['label_text']]
    sst[splitname] = sst[splitname].add_column('gold_label', dist)
    sst[splitname] = sst[splitname].add_column('sentence', sst[splitname]['text'])

print_label_dist(sst)


def r_get_batch_token_ids(batch, tokenizer):
    """Map `batch` to a tensor of ids. The return
    value should meet the following specification:

    1. The max length should be 512.
    2. Examples longer than the max length should be truncated
    3. Examples should be padded to the max length for the batch.
    4. The special [CLS] should be added to the start and the special
       token [SEP] should be added to the end.
    5. The attention mask should be returned
    6. The return value of each component should be a tensor.

    Parameters
    ----------
    batch: list of str
    tokenizer: Hugging Face tokenizer

    Returns
    -------
    dict with at least "input_ids" and "attention_mask" as keys,
    each with Tensor values

    """
    pass
    ##### YOUR CODE HERE
    
    encoded_inputs = tokenizer.batch_encode_plus(
        batch,
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length=512,           # Set maximum length to 512
        truncation=True,          # Truncate longer sequences
        padding='max_length',     # Pad the sequences to max length
        return_tensors='pt'       # Return PyTorch tensors
    )

    return encoded_inputs
    

class RobertaClassifierModule(nn.Module):
    def __init__(self,
            n_classes,
            hidden_activation,
            hidden_dim1, hidden_dim2,
            weights_name="roberta-base"):
        """This module loads a Transformer based on  `weights_name`,
        puts it in train mode, add a dense layer with activation
        function give by `hidden_activation`, and puts a classifier
        layer on top of that as the final output. The output of
        the dense layer should have the same dimensionality as the
        model input.

        Parameters
        ----------
        n_classes : int
            Number of classes for the output layer
        hidden_activation : torch activation function
            e.g., nn.Tanh()
        weights_name : str
            Name of pretrained model to load from Hugging Face

        """
        super().__init__()
        self.n_classes = n_classes
        self.weights_name = weights_name
        self.bert = AutoModel.from_pretrained(self.weights_name)
        self.bert.train()
        self.hidden_activation = hidden_activation
        self.hidden_dim = self.bert.embeddings.word_embeddings.embedding_dim
        self.hidden_dim1= hidden_dim1
        self.hidden_dim2= hidden_dim2
        self.pooling = nn.AdaptiveAvgPool1d(1)
        # Add the new parameters here using `nn.Sequential`.
        # We can define this layer as
        #
        #  h = f(cW1 + b_h)
        #  y = hW2 + b_y
        #
        # where c is the final hidden state above the [CLS] token,
        # W1 has dimensionality (self.hidden_dim, self.hidden_dim),
        # W2 has dimensionality (self.hidden_dim, self.n_classes),
        # f is the hidden activation, and we rely on the PyTorch loss
        # function to add apply a softmax to y.
        self.classifier_layer = None
        ##### YOUR CODE HERE
        self.classifier_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),  # First linear layer
            self.hidden_activation,  # Activation function
            nn.Linear(self.hidden_dim, self.hidden_dim1),
            self.hidden_activation,
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            self.hidden_activation,
            nn.Linear(self.hidden_dim2, n_classes)  # Output layer to number of classes
        )
        '''
        self.model = BertClassifierModule(
            n_classes=self.n_classes_,
            hidden_activation=self.hidden_activation,
            weights_name=self.weights_name
        )
        '''


    def forward(self, indices, mask):
        """Process `indices` with `mask` by feeding these arguments
        to `self.bert` and then feeding the initial hidden state
        in `last_hidden_state` to `self.classifier_layer`

        Parameters
        ----------
        indices : tensor.LongTensor of shape (n_batch, k)
            Indices into the `self.bert` embedding layer. `n_batch` is
            the number of examples and `k` is the sequence length for
            this batch
        mask : tensor.LongTensor of shape (n_batch, d)
            Binary vector indicating which values should be masked.
            `n_batch` is the number of examples and `k` is the
            sequence length for this batch

        Returns
        -------
        tensor.FloatTensor
            Predicted values, shape `(n_batch, self.n_classes)`

        """
        pass
        ##### YOUR CODE HERE
        #outputs = self.bert(input_ids=indices, attention_mask=mask)
        #cls_output = outputs.last_hidden_state[:, 0, :]

        outputs = self.bert(input_ids=indices, attention_mask=mask)
        last_hidden_state = outputs.last_hidden_state
        # Pooling
        pooled_output = self.pooling(last_hidden_state.permute(0, 2, 1)).squeeze(-1)
        logits = self.classifier_layer(pooled_output)

        # Pass [CLS] token's embeddings through the classifier layer
        #logits = self.classifier_layer(cls_output)

        return logits

class RobertaClassifier(TorchShallowNeuralClassifier):
    def __init__(self,  weights_name, hidden_dim1=50, hidden_dim2=50, *args, **kwargs):
        self.weights_name = weights_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.weights_name)
        super().__init__(*args, **kwargs)
        self.params += ['weights_name']
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

    def build_graph(self):
        return RobertaClassifierModule(
            self.n_classes_, self.hidden_activation,self.hidden_dim1, self.hidden_dim2, self.weights_name)

    def build_dataset(self, X, y=None):
        data = r_get_batch_token_ids(X, self.tokenizer)
        if y is None:
            dataset = torch.utils.data.TensorDataset(
                data['input_ids'], data['attention_mask'])
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            y = [class2index[label] for label in y]
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(
                data['input_ids'], data['attention_mask'], y)
        return dataset

def processImbdCsv():
    # Load the data
    imdb_df = pd.read_csv("./cs224u/imdb.csv")

    # Function to clean HTML tags from reviews
    def clean_text(text):
        # Remove HTML tags using regex
        return re.sub(r'<.*?>', '', text)

    # Clean the 'review' column and rename it to 'sentence'
    imdb_df['sentence'] = imdb_df['review'].apply(clean_text)
    print("Check 'sentence' column addition:", imdb_df.head())

    # Map 'sentiment' to 'gold-label'
    #sentiment_mapping = {'positive': 1, 'negative': 0}  # Adjust mapping as needed
    imdb_df['gold_label'] = imdb_df['sentiment']
    print("Check 'gold_label' column addition:", imdb_df.head())

    # Drop the original 'review' and 'sentiment' columns
    imdb_df.drop(columns=['review', 'sentiment'], inplace=True)

    # Save the preprocessed data to a new CSV file (optional)
    imdb_df.to_csv("./cs224u/processed_imdb.csv", index=False)

    # Display the first few rows of the processed DataFrame
    print(imdb_df.head())


roberta_finetune = RobertaClassifier(
    weights_name = "roberta-base",
    hidden_activation=nn.ReLU(),
    eta=0.05,          # Low learning rate for effective fine-tuning.
    batch_size=4,         # Small batches to avoid memory overload.
    gradient_accumulation_steps=4,  # Increase the effective batch size to 32.
    early_stopping=True,  # Early-stopping
    n_iter_no_change=4)   # params.


#train on dynasent data
_ = roberta_finetune.fit(
    train_data['sentence'],
    train_data['gold_label'])

# Process the data for fine tuning
processImbdCsv()

# Load the CSV file into a DataFrame
imdb_data = pd.read_csv('/home/ubuntu/ML/ML/cs224u/processed_imdb.csv')

print(imdb_data.head())
print(imdb_data['sentence'].head())

imdb_sentences = imdb_data['sentence'].to_list() 
imdb_labels = imdb_data['gold_label'].to_list()

# Get 1000 random indices
sample_size = 1000
random_indices = random.sample(range(len(imdb_sentences)), sample_size)

# Sample 1000 entries from both imdb_sentences and imdb_labels
sampled_sentences = [imdb_sentences[i] for i in random_indices]
sampled_labels = [imdb_labels[i] for i in random_indices]

print("Sampled Sentences:", sampled_sentences[:5])  # Printing first 5 as a preview
print("Sampled Labels:", sampled_labels[:5])

_ = roberta_finetune.fit(sampled_sentences, sampled_labels)

'''
Roberta_finetune.save_pretrained('./fine_tuned_roberta')

tokenizer.save_pretrained('./fine_tuned_roberta')
'''

'''
_ = bert_finetune.fit(
    dynasent_r1['train']['sentence'],
    dynasent_r1['train']['gold_label'])
'''

#prediction on the samples
preds = roberta_finetune.predict(validation_data['sentence'])

print(classification_report(validation_data['gold_label'], preds, digits=3))

#preds = bert_finetune.predict(dynasent_r1['validation']['sentence'])

#preds = bert_finetune.predict(test_data['sentence'])

#print(classification_report(dynasent_r1['validation']['gold_label'], preds, digits=3))

#print(classification_report(dynasent_r1['validation']['gold_label'], preds, digits=3))

#preds = bert_finetune.predict(sst['validation']['sentence'])

#print(classification_report(sst['validation']['gold_label'], preds, digits=3))

#preds = bert_finetune.predict(dynasent_r1['validation']['sentence'])

#print(classification_report(dynasent_r1['validation']['gold_label'], preds, digits=3))


if not os.path.exists(os.path.join("data", "sentiment", "cs224u-sentiment-test-unlabeled.csv")):
    os.makedirs(os.path.join('data', 'sentiment'), exist_ok=True)
    wget.download('https://web.stanford.edu/class/cs224u/data/cs224u-sentiment-test-unlabeled.csv', out='data/sentiment/')

bakeoff_df = pd.read_csv("https://web.stanford.edu/class/cs224u/data/cs224u-sentiment-test-unlabeled.csv")
print(bakeoff_df.head())

def predict_labels(text_list):
    # Your method to get predictions from the model
    # This function should return a list of predictions corresponding to the input text list
    return roberta_finetune.predict(text_list)

bakeoff_df['prediction'] = predict_labels(bakeoff_df['sentence'].tolist())

bakeoff_df.to_csv("data/sentiment/cs224u-sentiment-bakeoff-entry.csv", index=False)
