# Clinical Notes Dataset Description

## Overview
This dataset contains de-identified free-text clinical notes collected 
from the MIMIC-IV database. Each entry in the dataset represents a 
clinical note with various sections, including chief complaints, 
major surgical or invasive procedures, history of present illness, 
and physical examination findings.

## Example of Data
Here is an example from the dataset:

- **Chief Complaint**: Right distal tibia fracture.
- **Major Surgical or Invasive Procedure**: ORIF right distal tibia 
fracture with removal of external fixator.
- **History of Present Illness**: MR is a ___ year old male who suffered 
a right distal tibia fracture after he jumped 2 stories fleeing the police. 
He underwent an external fixator placement with closed reduction to his 
right distal tibia on ___. He now presents for definitive fixation.
- **Physical Exam**: Upon admission, alert and oriented. Cardiac: 
Regular rate rhythm. Chest: Lungs clear bilaterally. 
Abdomen: Soft non-tender non-distended. Extremities: RLE ex-fix intact, 
+pulses/sensation.
- **Label**: Home (There are 4 unique labels: Home, 
Home with Service, Extended Care, Expired)

## File Format
The data is split into three sets:
- Training Set (80%)
- Test Set (20%)
- Note: We do not have a development set because 
all the models we tested are language models (BERT, GPT), 
and there are not many hyperparameters to be tuned, 
making it unnecessary to include a dev set.

Each set is in CSV format:
1. `train_labels.csv`: Contains the training clinical note.
2. `train_texts,csv`: Training label for the train texts.
2. `test_texts.csv`: Contains the testing clinical note.
2. `test_labels.csv`: The assigned label for the test texts.

## Data Source
The data is sourced from the MIMIC-IV database, which is a publicly available dataset of de-identified clinical notes. More information about the dataset can be found at the PhysioNet website: [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/).
