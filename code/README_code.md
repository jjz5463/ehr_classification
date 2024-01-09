# Run Baselines and Extension 1 

This file contains command-line examples to execute the basic baseline (RoBERTa w/o finetuning),
strong baseline (RoBERTa w/ finetuning), and Extension 1 (PubMedBERT) from the command line.

## Change Directory Before Execution

All the commands below assume that you are in the directory where this `README.md` file is located.

```bash
cd <path_where_you_saved_the_final_code_folder>/final_code/code
```

## Basic Baseline Execution Instructions

Instructions for running the basic baseline, which is RoBERTa without finetuning.

### Example Command

To execute the basic baseline, use the following command:

```bash
python baselines/baselines.py --method basic --train_texts ../data/train_texts.csv --train_labels ../data/train_labels.csv --test_texts ../data/test_texts.csv --pred_output ../output/basic_baseline_pred.csv
```

### Output

 * The result of the prediction will be stored at the path specified after `--pred_output`.
 * Using the command provided above, the prediction file will be saved under `ouput/` as `basic_baseline_pred.csv`

## Strong Baseline Execution Instructions

Instructions for running the strong baseline, which utilizes RoBERTa with fine-tuning.

### Example Command

To execute the strong baseline, use the following command:

```bash
python baselines/baselines.py --method fancy --train_texts ../data/train_texts.csv --train_labels ../data/train_labels.csv --test_texts ../data/test_texts.csv --pred_output ../output/fancy_baseline_pred.csv
```

### Output

 * The result of the prediction will be stored at the path specified after `--pred_output`.
 * Using the command provided above, the prediction file will be saved under `ouput/` as `fancy_baseline_pred.csv`

## Extension 1 - PubMedBERT for Patient Classification

Utilize PubMedBERT, a BERT variant pre-trained on medical domain data from PubMed.

### Overview

PubMedBERT maintains the same architecture as the original BERT but is re-pretrained using data from PubMed, making it more suited for medical text analysis.

We have implemented two versions of the classification head:
1. A simple linear classifier (softmax).
2. A bidirectional LSTM that feeds into a linear classifier.

### Running the Code

You can run PubMedBERT with different heads using the following commands:

1. **For Linear Head:**

   ```bash
   python extension1/pubmed.py --head linear --train_texts ../data/train_texts.csv --train_labels ../data/train_labels.csv --test_texts ../data/test_texts.csv --pred_output ../output/pubmed_linear_pred.csv
   ```

2. **For LSTM Head:**

   ```bash
   python extension1/pubmed.py --head lstm --train_texts ../data/train_texts.csv --train_labels ../data/train_labels.csv --test_texts ../data/test_texts.csv --pred_output ../output/pubmed_lstm_pred.csv
   ```
   
### Output

 * The result of the prediction will be stored at the path specified after `--pred_output`.
 * Using the command provided above, the prediction file will be saved under
`ouput/` as `pubmed_linear_pred.csv` and `pubmed_lstm_pred.csv`

## Extension 2 - GPT 3.5 Turbo In-Context learning

 * In `extension2/incontext_learning.ipynb`, we implement in context learning on GPT 3.5 Turbo
using opein ai api. We tested zero-shot, zero shot with explaination of meaning of label,
two-shot learning, and three shot learning.
 * You have to supply your own openai api to run the juypter notebook
 * This part cannot be run using command line
 * The output of 4 in-context learning variant is pre-loaded under `output/` as
   * `zero_shot_predict.csv`
   * `zero_shot_explain_predict.csv`
   * `two_shot_predict.csv`
   * `three_shot_predict.csv`
 * Note: Examples in the 2-shot and 3-shot scenarios were drawn from 
the training dataset, and all predictions were made on the same test 
set as the BERT-related models.