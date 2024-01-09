import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_metric
from sklearn.preprocessing import LabelEncoder

# Command line argument parsing
parser = argparse.ArgumentParser(description='Run the Roberta model with specified configuration.')
parser.add_argument('--method', choices=['basic', 'fancy'], help='Configuration mode: basic or fancy')
parser.add_argument('--train_texts', type=str, help='Path to the train_texts.csv file')
parser.add_argument('--train_labels', type=str, help='Path to the train_labels.csv file')
parser.add_argument('--test_texts', type=str, help='Path to the test_texts.csv file')
parser.add_argument('--pred_output', type=str, help='Path to the pred.csv file')
args = parser.parse_args()

# Read data from the provided file paths
train_texts = pd.read_csv(args.train_texts)['text'].tolist()
train_labels = pd.read_csv(args.train_labels)['label'].tolist()
test_texts = pd.read_csv(args.test_texts)['text'].tolist()

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
train_data = (train_texts, train_labels)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def roberta(train_data, test_texts, num_epochs=3, freeze_base_model=True):
    # Unpack training data
    train_texts, train_labels = train_data
    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(set(train_labels)))
    # Freeze the base model's parameters if required
    if freeze_base_model:
        for param in model.roberta.parameters():
            param.requires_grad = False
    # Prepare training dataset
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./checkpoints',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        #logging_steps=10,
    )
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        #compute_metrics=compute_metrics
    )
    # Train the model
    trainer.train()
    # Prepare test texts for prediction
    test_dataset = TextDataset(test_texts, [0]*len(test_texts), tokenizer)  
    # Get predictions
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    return preds

# Adjust the call to roberta function based on the command line argument
freeze_base_model = True if args.method == 'basic' else False
pred = roberta(train_data, test_texts, num_epochs=3, freeze_base_model=freeze_base_model)
pred = label_encoder.inverse_transform(pred)

# Save predictions to a CSV file
df = pd.DataFrame(pred, columns=['Prediction'])
df.to_csv(args.pred_output, index=False)
