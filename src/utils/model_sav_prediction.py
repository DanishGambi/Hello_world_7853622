import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors


def smile_to_fingerprint(input_data):
    features_data = []

    df = pd.DataFrame(
        {
            "smiles": [input_data]
        }
    )

    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            features_data.append({
                'MW': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'RB': Descriptors.NumRotatableBonds(mol),
                'Atoms': mol.GetNumAtoms(),
                'HeavyAtoms': mol.GetNumHeavyAtoms(),
                'Aromatic': sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()),
                'Charged': sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() != 0)
            })
        else:
            features_data.append({})  # Пустой словарь для невалидных SMILES

    # Создать чистый DataFrame
    features_df = pd.DataFrame(features_data)

    return features_df

def load_model(path_model: str):
    loaded_model = pickle.load(open(path_model, 'rb'))

def predict_sav(smile: str):
    X_df = smile_to_fingerprint(smile)

