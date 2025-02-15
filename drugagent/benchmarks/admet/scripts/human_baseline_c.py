import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os
import torch.nn as nn
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Change the current directory to the 'env' folder located in the same parent folder as the script
current_folder = os.path.abspath(os.path.dirname(__file__))  # Current script directory
parent_folder = os.path.abspath(os.path.join(current_folder, '..'))  # Parent directory
env_folder = os.path.join(parent_folder, 'env')  # Path to the 'env' folder

# Switch to the 'env' directory
os.chdir(env_folder)

# Load ChemBERTa model and tokenizer
model_path = "DeepChem/ChemBERTa-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
model._modules["lm_head"] = nn.Identity()  # Remove the language model head

def smiles2chembertemb(smiles, mode='cls'):
    """
    Generates a ChemBERTa embedding for a given SMILES string.
    
    Args:
        smiles (str): A SMILES string representing the molecular structure.
        mode (str): The embedding mode, either 'cls' for the CLS token embedding or 'mean' for the mean of all token embeddings.
    
    Returns:
        list: A list of floats representing the ChemBERTa embedding.
    """
    try:
        encoded_input = tokenizer(smiles, return_tensors="pt")
        model_output = model(**encoded_input)
        
        if mode == 'cls':
            embedding = model_output[0][:, 0, :]  # CLS token embedding
        elif mode == 'mean':
            embedding = torch.mean(model_output[0], dim=1)  # Mean pooling of all tokens
        else:
            raise ValueError("Unsupported mode. Choose 'cls' or 'mean'.")
        
        return np.array(embedding.squeeze().tolist())
    except Exception as e:
        print(f'Error processing SMILES {smiles}: {str(e)}')
        return np.zeros(768)  # Default to zero vector of the expected size

# Load train, validation, and test datasets
train_data = pd.read_csv("train.csv")
val_data = pd.read_csv("valid.csv")
test_data = pd.read_csv("test.csv")

print(f"Loaded {len(train_data)} training samples, {len(val_data)} validation samples, and {len(test_data)} test samples.")

X_train_smiles = train_data['Drug']
y_train = train_data['Y']
X_val_smiles = val_data['Drug']
y_val = val_data['Y']

# ***********************************************
# Computing ChemBERTa embeddings for train and validation datasets
# ***********************************************
X_train_chemberta = np.array([smiles2chembertemb(sm) for sm in X_train_smiles])
X_val_chemberta = np.array([smiles2chembertemb(sm) for sm in X_val_smiles])

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train_chemberta, y_train)

# Make predictions and calculate ROC AUC scores
train_preds = rf_model.predict_proba(X_train_chemberta)[:, 1]
val_preds = rf_model.predict_proba(X_val_chemberta)[:, 1]

train_roc_auc = roc_auc_score(y_train, train_preds)
valid_roc_auc = roc_auc_score(y_val, val_preds)

print("Train ROC AUC Score: " + str(train_roc_auc))
print("Validation ROC AUC Score: " + str(valid_roc_auc))

X_test_smiles = test_data['Drug']

# Compute ChemBERTa embeddings for test data
X_test_chemberta = np.array([smiles2chembertemb(sm) for sm in X_test_smiles])

# Generate prediction probabilities
test_preds = rf_model.predict_proba(X_test_chemberta)[:, 1]

test_roc_auc = roc_auc_score(test_data["Y"], test_preds)

print("Test ROC AUC Score: " + str(test_roc_auc))

# Add predictions to the test set and save submission
test_data['Predicted'] = test_preds
output_file = "submission.csv"
test_data.to_csv(output_file, index=False)

print(f"Submission file saved to {output_file}.")