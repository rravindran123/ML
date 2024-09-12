import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import classification_report
import torch


dynasent_r1 = load_dataset("dynabench/dynasent", 'dynabench.dynasent.r1.all', trust_remote_code=True)
print(dynasent_r1)

validation_data = dynasent_r1['validation'].shuffle(seed=42).select(range(1000))

dynasent_r2 = load_dataset("dynabench/dynasent", 'dynabench.dynasent.r2.all')
print(dynasent_r2)

validation_data_2 = dynasent_r2['validation'].shuffle(seed=42).select(range(720))


def validation(data):

    sentences = data['sentence']#.tolist()

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('/home/ubuntu/ML/fine-tuned-bert')

    # Tokenize the sentences
    encoded_inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt')

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

    print(classification_report(data['gold_label'], predicted_labels, digits=3))


# Assuming your sentences are in a column named 'sentence'
#sentences = validation_data['sentence']#.tolist()

validation(validation_data)

#sentences_2 = validation_data_2['sentence']

validation(validation_data_2)



# Add predictions to the DataFrame
#data['predictions'] = predicted_labels

# Save the DataFrame with predictions
#data.to_csv('data/sentiment/cs224u-sentiment-bakeoff-entry.csv', index=False)



