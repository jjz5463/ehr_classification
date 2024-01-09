import pandas as pd
import numpy as np
import torch
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_metric
from sklearn.preprocessing import LabelEncoder

import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run BERT model with different heads.')
parser.add_argument('--head', type=str, choices=['lstm', 'linear'], required=True,
                    help='Type of model head to use (lstm or linear)')
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

# Check if GPU is available and set the device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class CustomBertModel(BertPreTrainedModel):
    def __init__(self, config, use_lstm=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.use_lstm = use_lstm

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        if self.use_lstm:
            self.lstm = nn.LSTM(config.hidden_size, 
                                config.hidden_size, 
                                batch_first=True,
                                bidirectional=True)
            self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.use_lstm:
            sequence_output = outputs[0]  # Get the sequence output
            lstm_output, _ = self.lstm(sequence_output)
            # Assuming we use the output of the last LSTM cell for classification
            lstm_output = lstm_output[:, 0, :]
            output = self.dropout(lstm_output)
        else:
            pooled_output = outputs[1]  # Get the pooled output
            output = self.dropout(pooled_output)

        logits = self.classifier(output)

        # Rest of the forward method remains the same...
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def custom_pubmedbert(train_data, test_texts, num_epochs=3, use_lstm=False):
    # Unpack data
    train_texts, train_labels = train_data

    # Load tokenizer and model specific to PubMedBERT
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    model = CustomBertModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', num_labels=len(set(train_labels)),use_lstm=use_lstm).to(device)

    # Prepare dataset
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    
    # Prepare test texts for prediction
    test_dataset = TextDataset(test_texts, [0]*len(test_texts), tokenizer) 

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Get predictions
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    return preds

# Run the model with the specified head
if args.head == 'lstm':
    preds = custom_pubmedbert(train_data, test_texts, num_epochs=3, use_lstm=True)
else:
    preds = custom_pubmedbert(train_data, test_texts, num_epochs=3, use_lstm=False)

preds = label_encoder.inverse_transform(preds)

# Save predictions to a CSV file
df = pd.DataFrame(preds, columns=['Prediction'])
df.to_csv(args.pred_output, index=False)