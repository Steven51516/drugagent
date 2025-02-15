import os
# Change the current directory to the 'env' folder located in the same parent folder as the script
current_folder = os.path.abspath(os.path.dirname(__file__))  # Current script directory
parent_folder = os.path.abspath(os.path.join(current_folder, '..'))  # Parent directory
env_folder = os.path.join(parent_folder, 'env')  # Path to the 'env' folder

# Switch to the 'env' directory
os.chdir(env_folder)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

# Function to convert SMILES to Morgan fingerprints
def smiles_to_fingerprint(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((nBits,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)

# Load train, validation, and test datasets
train_data = pd.read_csv("train.csv")
val_data = pd.read_csv("valid.csv")
test_data = pd.read_csv("test.csv")

print(f"Loaded {len(train_data)} training samples, {len(val_data)} validation samples, and {len(test_data)} test samples.")

# Example: Print the first few rows of the training data
print(train_data.head())

# Convert SMILES to fingerprints for training, validation, and test sets
X_train_fp = np.array([smiles_to_fingerprint(smiles) for smiles in train_data['Drug']])
X_val_fp = np.array([smiles_to_fingerprint(smiles) for smiles in val_data['Drug']])
X_test_fp = np.array([smiles_to_fingerprint(smiles) for smiles in test_data['Drug']])

y_train = train_data['Y']
y_val = val_data['Y']

# ***********************************************
# Model training and ROC AUC score calculation
# ***********************************************
# Train RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_fp, y_train)

# Predict probabilities
train_probs = clf.predict_proba(X_train_fp)[:, 1]
val_probs = clf.predict_proba(X_val_fp)[:, 1]

# Calculate ROC AUC scores
train_roc_auc = roc_auc_score(y_train, train_probs)
valid_roc_auc = roc_auc_score(y_val, val_probs)

# ***********************************************
# End of the main training module
# ***********************************************

print("Train ROC AUC Score: " + str(train_roc_auc))
print("Validation ROC AUC Score: " + str(valid_roc_auc))

# Use the model to predict probabilities on the test set
test_preds = clf.predict_proba(X_test_fp)[:, 1]
# Predict on the test set
# test_preds = model.predict_proba(test_descriptors_scaled)[:, 1]
test_roc_auc = roc_auc_score(test_data['Y'], test_preds)


print("Test ROC AUC Score: " + str(test_roc_auc))