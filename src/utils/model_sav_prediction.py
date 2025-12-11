import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors
import sklearn


async def smile_to_fingerprint(input_data):
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

        features_df['MW_violation'] = (features_df['MW'] > 500).astype(int)  # Нарушение правила Липински
        features_df['LogP_violation'] = (features_df['LogP'] > 5).astype(int)  # Нарушение правила Липински
        features_df['HBD_violation'] = (features_df['HBD'] > 5).astype(int)  # Нарушение правила Липински
        features_df['HBA_violation'] = (features_df['HBA'] > 10).astype(int)  # Нарушение правила Липински

        # Drug-likeness - КЛЮЧЕВОЙ КАТЕГОРИАЛЬНЫЙ ПРИЗНАК
        features_df['is_druglike'] = ((features_df['MW'] <= 500) &
                                      (features_df['LogP'] <= 5) &
                                      (features_df['HBD'] <= 5) &
                                      (features_df['HBA'] <= 10)).astype(int)

        # Структурные бинарные признаки
        features_df['has_aromatic'] = (features_df['Aromatic'] > 0).astype(int)
        features_df['has_charge'] = (features_df['Charged'] > 0).astype(int)

        # Б) КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ С НЕСКОЛЬКИМИ КАТЕГОРИЯМИ
        # Размер молекулы (4 категории)
        features_df['size_category'] = pd.cut(features_df['MW'],
                                              bins=[0, 250, 400, 600, 1000],
                                              labels=['small', 'medium', 'large', 'very_large'],
                                              include_lowest=True)

        # Липофильность (3 категории)
        features_df['logp_category'] = pd.cut(features_df['LogP'],
                                              bins=[-10, 1, 3, 10],
                                              labels=['hydrophilic', 'moderate', 'lipophilic'],
                                              include_lowest=True)

        # Количество доноров H-связей (порядковая)
        features_df['hbd_category'] = pd.cut(features_df['HBD'],
                                             bins=[-1, 0, 2, 5, 10],
                                             labels=['none', 'few', 'moderate', 'many'],
                                             include_lowest=True)

        # Полярность (4 категории)
        features_df['polarity_category'] = pd.cut(features_df['TPSA'],
                                                  bins=[0, 60, 120, 200, 300],
                                                  labels=['nonpolar', 'moderate', 'polar', 'highly_polar'],
                                                  include_lowest=True)

        # С) ДОПОЛНИТЕЛЬНЫЙ ПРИЗНАК - сумма нарушений правила Липински
        features_df['ro5_violations'] = (features_df['MW_violation'] +
                                         features_df['LogP_violation'] +
                                         features_df['HBD_violation'] +
                                         features_df['HBA_violation'])

        return features_df

async def load_model(path_model: str):
    loaded_model = pickle.load(open(path_model, 'rb'))

    return loaded_model

async def predict_sav(smile: str, path_model:str):
    X_df = await smile_to_fingerprint(smile)
    model = await load_model(path_model)

    predict = model.predict(X_df)

    return predict
