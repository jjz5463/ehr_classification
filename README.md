# Patients Discharge Disposition Classification

The objective of this project is to classify free text from the MIMIC-IV discharge data, 
which documents key information about ICU patients at the time of their discharge, into 
three categories of discharge disposition: Home, Extended Care, and Expired. To achieve this, 
the project employed several BERT models, including a basic RoBERTa without fine-tuning, 
RoBERTa with fine-tuning, PubMedBERT with a linear head layer and fine-tuning, PubMedBERT with 
a bi-directional LSTM head layer and fine-tuning, and GPT-3.5 Turbo in zero-shot, two-shot, 
and three-shot in-context learning scenarios.

## Project Folder Structure

This document outlines the structure and contents of the project folder. 
The project folder is organized into three major subfolders:

- data/
- code/
- output/

## Code Folder

- Located in the `code` subfolder, you will find four subfolders: `baselines`, `extension1`, 
`extension2`, `evaluation`.
- Inside `baselines` folder, you will find `baselines.py`, which contains both simple 
and strong baseline models.
- Inside `extension1` folder, there is `pubmed.py`, which contains pubmed bert model
- To execute above models from command line, refer to the commands provided in `README_code.md`.
- Inside the `extension2` folder, there is a Jupyter Notebook file named 
`incontext_learning.ipynb`, which uses the OpenAI API to perform in-context learning.
Please use a Jupyter-compatible IDE or Google Colab to access it.
- Inside `evaluation` folder, there is `scoring.py`, which is evaluation scripts.
- To execute evaluation from command line, refer to `output/REAMD_scoring.md` under the output/ folder

## Data Folder

- The `data` folder is where all test and train texts and labels are stored.
- To learn more about data sources, refer to `README_data.md`

## Output Folder

- Predicted labels generated from the baseline models, pubmed model, and GPT 
are also stored in this directory.
- `README_scoring.md` shows the command line on how to run your evaluation script on the output.


## Execution Flow

1. After running one of the models, its output prediction will be store under `output/` folder
2. You can execute `scoring.py` to evalaute output using the commands specified in `README_scoring.md`.

## Setup Instructions

Before starting, it is recommended to set up a new conda environment and install the necessary dependencies:

```bash
conda create -n project python=3.10
conda activate project
pip install -r requirements.txt
```




