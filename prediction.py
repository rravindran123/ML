import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# Path to the CSV file
file_path = 'https://web.stanford.edu/class/cs224u/data/cs224u-sentiment-test-unlabeled.csv'

# Load the CSV into a DataFrame
data = pd.read_csv(file_path)

# Assuming your sentences are in a column named 'sentence'
sentences = data['sentence'].tolist()

from transformers import RobertaTokenizer

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('/home/ubuntu/ML/fine-tuned-bert')

# Tokenize the sentences
encoded_inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt')

from transformers import RobertaForSequenceClassification
import torch

# Load the model
model = BertForSequenceClassification.from_pretrained('/home/ubuntu/ML/fine-tuned-bert')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

with torch.no_grad():
    outputs = model(**encoded_inputs.to(device))

# Get the predicted class indices
predictions = torch.argmax(outputs.logits, dim=-1)

# Map indices to class names if you have a mapping available
class_names = ['neutral', 'positive', 'negative']  # Adjust as per your classes
predicted_labels = [class_names[pred] for pred in predictions]

# Add predictions to the DataFrame
data['predictions'] = predicted_labels

# Save the DataFrame with predictions
data.to_csv('data/sentiment/cs224u-sentiment-bakeoff-entry.csv', index=False)



