from .protein_utils import *
from .drug_utils import *
from tdc.chem_utils.featurize.molconvert import MoleculeFingerprint
import numpy as np

class DrugFeaturizer:
    def __init__(self, method="ECFP"):
        self.method = method
        self.mol_fingerprint = None

        supported_methods = [
            "dgl", "pyg", "onehot", "chemberta",
            "Morgan", "ECFP0", "ECFP2", "ECFP4", "ECFP6",
            "MACCS", "PubChem", "Daylight"
        ]
        # Predefined transformation methods
        if method == "dgl":
            self.transform_func = smiles2DGL
        elif method == "pyg":
            self.transform_func = smiles2PyG
        elif method == "onehot":
            self.transform_func = smiles2onehot
        elif method == "chemberta":
            self.transform_func = smiles2chemberta
        elif method == "Morgan" or method.startswith("ECFP") or method == "MACCS" or method == "PubChem" or method == "Daylight":
            self.mol_fingerprint = MoleculeFingerprint(fp=self.method)
            self.transform_func = self.smiles2TDCFingerprint
        else:
            raise ValueError(
                f"Unsupported method: {method}. Supported methods are: {', '.join(supported_methods)}"
            )


    def smiles2TDCFingerprint(self, smiles):
        """
        Use TDC MoleculeFingerprint to generate molecular fingerprints.
        Lazy initialization of the MoleculeFingerprint object is used to avoid redundant reinitialization.
        """
    
        return self.mol_fingerprint(smiles)


    def __call__(self, smiles):
        if isinstance(smiles, (list, tuple, np.ndarray)):  # If the input is a list-like object
            unique_values = np.unique(smiles)
            unique_transformed = [self.transform_func(str(item)) for item in unique_values]  # Transform each unique value
            mapping = dict(zip(unique_values, unique_transformed))  # Map unique values to their transformed results
            return [mapping[item] for item in smiles]  # Apply the mapping to the input
        else:
            return self.transform_func(smiles)



class ProteinFeaturizer:
    def __init__(self, method="aac"):

        supported_methods = [
            "aac", "quasi", "ct", "onehot", "esm"
        ]
        if method == "quasi":
            self.transform_func = protein2quasi
        elif method == "aac":
            self.transform_func = protein2aac
        elif method == "ct":
            self.transform_func = protein2ct
        elif method == "onehot":
            self.transform_func = protein2onehot
        elif method == "esm":
            self.transform_func = compute_esm_embedding
        else:
            raise ValueError(
                f"Unsupported method: {method}. Supported methods are: {', '.join(supported_methods)}"
            )

    def __call__(self, sequence):
        if isinstance(sequence, (list, tuple, np.ndarray)):  # If the input is a list-like object
            unique_values = np.unique(sequence)  # Find unique sequences
            unique_transformed = [self.transform_func(item) for item in unique_values]  # Transform each unique value
            mapping = dict(zip(unique_values, unique_transformed))  # Map unique values to their transformed results
            return [mapping[item] for item in sequence]  # Apply the mapping to the input
        else:
            return self.transform_func(sequence)
        

        

if __name__ == "__main__":
    # Test for DrugFeaturizer
    drug_featurizer = DrugFeaturizer(method="dgl")
    single_smiles = "CCO"
    list_smiles = ["CCO", "CCC", "CCO"]
    print("DrugFeaturizer single input:", drug_featurizer(single_smiles))
    print("DrugFeaturizer list input:", drug_featurizer(list_smiles))

    # Test for ProteinFeaturizer
    protein_featurizer = ProteinFeaturizer(method="aac")
    single_protein = "MKTAYIAKQRQISFVKSHFSRQ"
    list_proteins = ["MKTAYIAKQRQISFVKSHFSRQ", "GLSDGEWQQVLNVWGK", "MKTAYIAKQRQISFVKSHFSRQ"]
    print("ProteinFeaturizer single input:", protein_featurizer(single_protein))
    print("ProteinFeaturizer list input:", protein_featurizer(list_proteins))