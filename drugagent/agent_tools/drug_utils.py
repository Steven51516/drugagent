import numpy as np
import torch
import torch.nn as nn
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from functools import partial
from torch_geometric.utils import from_dgl
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Lazy initialization containers
_lazy_initializations = {
    "node_featurizer": None,
    "edge_featurizer": None,
    "transform_func": None,
    "onehot_enc": None,
    "tokenizer": None,
    "model": None
}

# Lazy initialization of featurizers
def get_node_featurizer():
    if _lazy_initializations["node_featurizer"] is None:
        _lazy_initializations["node_featurizer"] = CanonicalAtomFeaturizer()
    return _lazy_initializations["node_featurizer"]

def get_edge_featurizer():
    if _lazy_initializations["edge_featurizer"] is None:
        _lazy_initializations["edge_featurizer"] = CanonicalBondFeaturizer(self_loop=True)
    return _lazy_initializations["edge_featurizer"]

def get_transform_func():
    if _lazy_initializations["transform_func"] is None:
        _lazy_initializations["transform_func"] = partial(
            smiles_to_bigraph,
            node_featurizer=get_node_featurizer(),
            edge_featurizer=get_edge_featurizer(),
            add_self_loop=True
        )
    return _lazy_initializations["transform_func"]

# SMILES to DGL graph
def smiles2DGL(smiles):
    """
    Transforms a SMILES string into a DGL graph representation with canonical atom and bond features.
    
    Args:
        smiles (str): A SMILES string representing the molecular structure of a drug.

    Returns:
        dgl.DGLGraph: A DGL graph object representing the molecule with node and edge features for input to graph neural networks.
    """
    transform_func = get_transform_func()
    graph = transform_func(smiles)
    return graph

# SMILES to PyG graph
def smiles2PyG(smiles):
    """
    Transforms a SMILES string into a PyTorch Geometric graph representation with canonical atom and bond features.
    
    Args:
        smiles (str): A SMILES string representing the molecular structure of a drug.

    Returns:
        torch_geometric.data.Data: A PyTorch Geometric graph object.
    """
    transform_func = get_transform_func()
    graph = transform_func(smiles)
    graph = from_dgl(graph)
    return graph

# One-hot encoder initialization
def get_onehot_encoder():
    if _lazy_initializations["onehot_enc"] is None:
        smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
                       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
                       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
                       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
                       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']
        _lazy_initializations["onehot_enc"] = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))
    return _lazy_initializations["onehot_enc"]

# SMILES to one-hot encoding
MAX_SEQ = 100
def smiles2onehot(x):
    onehot_enc = get_onehot_encoder()
    smiles_char = onehot_enc.categories_[0].tolist()
    temp = list(x)
    temp = [i if i in smiles_char else '?' for i in temp]
    if len(temp) < MAX_SEQ:
        temp = temp + ['?'] * (MAX_SEQ - len(temp))
    else:
        temp = temp[:MAX_SEQ]
    return onehot_enc.transform(np.array(temp).reshape(-1, 1)).toarray().T

# ChemBERTa model and tokenizer initialization
def get_tokenizer_and_model():
    if _lazy_initializations["tokenizer"] is None:
        model_path = "DeepChem/ChemBERTa-77M-MTR"
        _lazy_initializations["tokenizer"] = AutoTokenizer.from_pretrained(model_path)
        _lazy_initializations["model"] = AutoModelForMaskedLM.from_pretrained(model_path)
        _lazy_initializations["model"]._modules["lm_head"] = nn.Identity()  # Remove the language model head
    return _lazy_initializations["tokenizer"], _lazy_initializations["model"]


def smiles2chemberta(smiles, mode='mean'):
    """
    Generates a ChemBERTa embedding for a given SMILES string.

    Args:
        smiles (str): A SMILES string representing the molecular structure.
        mode (str): The embedding mode, either 'cls' for the CLS token embedding or 'mean' for the mean of all token embeddings.

    Returns:
        list: A list of floats representing the ChemBERTa embedding.
    """
    try:
        tokenizer, model = get_tokenizer_and_model()
        # Tokenize the input SMILES string
        encoded_input = tokenizer(smiles, return_tensors="pt")
        # Get model output
        model_output = model(**encoded_input)
        
        # Extract embedding based on mode
        if mode == 'cls':
            embedding = model_output[0][:, 0, :]  # CLS token embedding
        elif mode == 'mean':
            embedding = torch.mean(model_output[0], dim=1)  # Mean pooling of all tokens
        else:
            raise ValueError("Unsupported mode. Choose 'cls' or 'mean'.")
        
        return np.array(embedding.squeeze().tolist())
    except Exception as e:
        print(f'Error processing SMILES {smiles}: {str(e)}')
        return []
