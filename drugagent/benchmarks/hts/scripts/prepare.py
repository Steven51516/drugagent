from tdc.single_pred import HTS
import pandas as pd

# Define task name and download directory
taskname = "admet-pampa-ncats"
download_dir = "../env"

# from tdc.single_pred import HTS
data = HTS(name = 'HIV')

# Get the default split
split = data.get_split()

# Save splits to CSV
train_data = pd.DataFrame(split['train'])
val_data = pd.DataFrame(split['valid'])
test_data = pd.DataFrame(split['test'])

# Save datasets locally
train_data.to_csv(f"{download_dir}/train.csv", index=False)
val_data.to_csv(f"{download_dir}/valid.csv", index=False)
test_data.to_csv(f"{download_dir}/test.csv", index=False)

print(f"{taskname} data prepared and saved to {download_dir}")