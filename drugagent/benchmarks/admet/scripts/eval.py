import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def get_score(submission_folder="../env"):
    # Paths to the submission
    submission_path = os.path.join(submission_folder, "submission.csv")
    
    # Read the submission file
    submission = pd.read_csv(submission_path)

    # Extract the 'Y' and 'Predicted' columns from the submission
    true_values = submission['Y'].tolist()
    predicted_values = submission['Predicted'].tolist()

    # Calculate the ROC-AUC score
    roc_auc = roc_auc_score(true_values, predicted_values)

    return roc_auc

if __name__ == "__main__":
    print(get_score())
