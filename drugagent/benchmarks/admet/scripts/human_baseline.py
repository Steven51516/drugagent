import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import os

# Change the current directory to the 'env' folder located in the same parent folder as the script
current_folder = os.path.abspath(os.path.dirname(__file__))  # Current script directory
parent_folder = os.path.abspath(os.path.join(current_folder, '..'))  # Parent directory
env_folder = os.path.join(parent_folder, 'env')  # Path to the 'env' folder

# Switch to the 'env' directory
os.chdir(env_folder)

# Function to compute ECFP fingerprints
def compute_ecfp(smiles_list, radius=2, n_bits=2048):
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fingerprints.append(np.array(fp))
        else:
            fingerprints.append(np.zeros(n_bits))  # Use zero vector for invalid SMILES
    return np.array(fingerprints)

# Load train, validation, and test datasets
train_data = pd.read_csv("train.csv")
val_data = pd.read_csv("valid.csv")
test_data = pd.read_csv("test.csv")

print(f"Loaded {len(train_data)} training samples, {len(val_data)} validation samples, and {len(test_data)} test samples.")

# Example: Print the first few rows of the training data
print(train_data.head())

X_train_smiles = train_data['Drug']
y_train = train_data['Y']

X_val_smiles = val_data['Drug']
y_val = val_data['Y']

# ***********************************************
# Computing ECFP fingerprints for train and validation datasets
# ***********************************************

X_train_ecfp = compute_ecfp(X_train_smiles)
X_val_ecfp = compute_ecfp(X_val_smiles)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train_ecfp, y_train)

# Make predictions and calculate ROC AUC scores
train_preds = rf_model.predict_proba(X_train_ecfp)[:, 1]
val_preds = rf_model.predict_proba(X_val_ecfp)[:, 1]

train_roc_auc = roc_auc_score(y_train, train_preds)
valid_roc_auc = roc_auc_score(y_val, val_preds)

# ***********************************************
# End of the main training module
# ***********************************************

print("Train ROC AUC Score: " + str(train_roc_auc))
print("Validation ROC AUC Score: " + str(valid_roc_auc))

X_test_smiles = test_data['Drug']

# Compute ECFP fingerprints for test data
X_test_ecfp = compute_ecfp(X_test_smiles)

# Generate prediction probabilities
test_preds = rf_model.predict_proba(X_test_ecfp)[:, 1]

test_roc_auc = roc_auc_score(test_data['Y'], test_preds)


print("Test ROC AUC Score: " + str(test_roc_auc))

# Add predictions to the test set and save submission
# test_data['Predicted'] = test_preds


# output_file = "submission.csv"
# test_data.to_csv(output_file, index=False)

# print(f"Submission file saved to {output_file}.")