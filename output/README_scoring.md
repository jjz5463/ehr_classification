# Scoring Script Instructions

This document provides instructions for running `scoring.py`, 
our evaluation script. We use five metrics 
(accuracy, f1, precision, recall, confusion matrix) 
to evaluate the performance of our 3-way classification. 
The three unique labels are: `Extended Care`, `Home`, and `Expired`.


Accuracy = \frac{T_p + T_n}{T_p+T_n+F_p+F_n}

Precision = \frac{T_p}{T_p+F_p}

Recall = \frac{T_p}/{T_p+T_n}

F_1 = 2*\frac{precision*recall}{precision+recall}

## Change Directory Before Execution

All the commands below assume that you are in the directory where this 
`README.md` file is located.

```bash
cd <path_where_you_saved_the_final_code_folder>/final_code/output
```

## Example Command

Use the following command format to run the script:
* basic baseline
```bash
python ../code/evaluation/scoring.py --pred_path basic_baseline_pred.csv --true_path ../data/test_labels.csv
```

* fancy baseline
```bash
python ../code/evaluation/scoring.py --pred_path fancy_baseline_pred.csv --true_path ../data/test_labels.csv
```

* pubmed linear head
```bash
python ../code/evaluation/scoring.py --pred_path pubmed_linear_pred.csv --true_path ../data/test_labels.csv
```

* pubmed lstm head
```bash
python ../code/evaluation/scoring.py --pred_path pubmed_lstm_pred.csv --true_path ../data/test_labels.csv
```

* gpt 3.5 turbo zero shot prediction
```bash
python ../code/evaluation/scoring.py --pred_path zero_shot_predict.csv --true_path ../data/test_labels.csv
```

* gpt 3.5 turbo zero shot with label explaination prediction
```bash
python ../code/evaluation/scoring.py --pred_path zero_shot_explain_predict.csv --true_path ../data/test_labels.csv
```

* gpt 3.5 turbo two shot prediction
```bash
python ../code/evaluation/scoring.py --pred_path two_shot_predict.csv --true_path ../data/test_labels.csv
```

* gpt 3.5 turbo three shot prediction
```bash
python ../code/evaluation/scoring.py --pred_path three_shot_predict.csv --true_path ../data/test_labels.csv
```

## sample output:

```makefile
accuracy: 0.7533333333333333
f1: 0.753996586918336
precision: 0.7608728368545381
recall: 0.7533333333333333
confusion_matrix:
251 61 5
14 483 96
1 119 170
```