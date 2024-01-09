import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluation(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix
    }

# Command line argument parsing
parser = argparse.ArgumentParser(description='Evaluate predictions against true labels.')
parser.add_argument('--pred_path', type=str, help='Path to the predictions CSV file')
parser.add_argument('--true_path', type=str, help='Path to the true labels CSV file')
args = parser.parse_args()

# Read data from the provided file paths
pred = pd.read_csv(args.pred_path)
test_labels = pd.read_csv(args.true_path)['label'].tolist()

# Perform evaluation
metrics = evaluation(test_labels, pred)
for metric, value in metrics.items():
    if metric != 'confusion_matrix':
        print(f'{metric}: {value}')
    else:
        print(f'{metric}:')
        for row in value:
            print(' '.join(str(x) for x in row))
