from .pybiomed_helper import CalculateAADipeptideComposition, \
CalculateConjointTriad, GetQuasiSequenceOrder
import numpy as np
import torch
from functools import lru_cache
from sklearn.preprocessing import OneHotEncoder


def protein2quasi(s):
	try:
		features = GetQuasiSequenceOrder(s)
	except:
		print('Quasi-seq fingerprint not working for smiles: ' + s + ' convert to 0 vectors')
		features = np.zeros((100, ))
	return np.array(features)

def protein2aac(s):
	try:
		features = CalculateAADipeptideComposition(s)
	except:
		print('AAC fingerprint not working for smiles: ' + s + ' convert to 0 vectors')
		features = np.zeros((8420, ))
	return np.array(features)


def protein2ct(s):
	try:
		features = CalculateConjointTriad(s)
	except:
		print('Conjoint Triad fingerprint not working for smiles: ' + s + ' convert to 0 vectors')
		features = np.zeros((343, ))
	return np.array(features)



amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
       'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']
MAX_SEQ_PROTEIN = 1000
enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))


def trans_protein(x):
	temp = list(x.upper())
	temp = [i if i in amino_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_PROTEIN:
		temp = temp + ['?'] * (MAX_SEQ_PROTEIN-len(temp))
	else:
		temp = temp [:MAX_SEQ_PROTEIN]
	return temp

def protein2onehot(x):
	return enc_protein.transform(np.array(trans_protein(x)).reshape(-1,1)).toarray().T


@lru_cache
def _load_esm_model(model_name: str = "esm1b_t33_650M_UR50S"):
    """
    Loads pre-trained FAIR ESM model from torch hub.

        *Biological Structure and Function Emerge from Scaling Unsupervised*
        *Learning to 250 Million Protein Sequences* (2019)
        Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth
        and Lin, Zeming and Liu, Jason and Guo, Demi and Ott, Myle and Zitnick,
        C. Lawrence and Ma, Jerry and Fergus, Rob


        *Transformer protein language models are unsupervised structure learners*
        2020 Rao, Roshan M and Meier, Joshua and Sercu, Tom and Ovchinnikov,
        Sergey and Rives, Alexander

    Pre-trained models:
    Full Name layers params Dataset Embedding Dim Model URL
    ========= ====== ====== ======= ============= =========
    ESM-1b   esm1b_t33_650M_UR50S 33 650M UR50/S 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt
    ESM1-main esm1_t34_670M_UR50S34 670M UR50/S 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50S.pt
    esm1_t34_670M_UR50D 34 670M UR50/D 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50D.pt
    esm1_t34_670M_UR100 34 670M UR100 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR100.pt
    esm1_t12_85M_UR50S 12 85M UR50/S 768 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t12_85M_UR50S.pt
    esm1_t6_43M_UR50S 6 43M UR50/S 768 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t6_43M_UR50S.pt

    :param model_name: Name of pre-trained model to load
    :type model_name: str
    :return: loaded pre-trained model
    """

    return torch.hub.load("facebookresearch/esm", model_name)

import os
import pickle
embed_dict = None
embed_file_path = os.path.join(os.path.dirname(__file__), "protein_to_esm_embedding.pkl")
if os.path.exists(embed_file_path):
    with open(embed_file_path, "rb") as f:
        embed_dict = pickle.load(f)


def compute_esm_embedding(
    sequence: str,
    representation: str = "sequence",
    model_name: str = "esm1b_t33_650M_UR50S",
    output_layer: int = 33,
) -> np.ndarray:
    
    if embed_dict is not None:
        # Check if the sequence exists in the dictionary
        if sequence in embed_dict:
            return embed_dict[sequence]

    if len(sequence) > 1022:
        sequence = sequence[:1022] # temp fix for long sequence

    model, alphabet = _load_esm_model(model_name)
    batch_converter = alphabet.get_batch_converter()

    data = [
        ("protein1", sequence),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(
            batch_tokens, repr_layers=[output_layer], return_contacts=True
        )
    token_representations = results["representations"][output_layer]

    if representation == "residue":
        return token_representations.numpy()

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first
    # residue is token 1.
    elif representation == "sequence":
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(
                token_representations[i, 1 : len(seq) + 1].mean(0)
            )
        return sequence_representations[0].numpy()
	
