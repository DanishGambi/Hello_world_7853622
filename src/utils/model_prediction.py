import h2o
from h2o import import_mojo, mojo_predict_pandas
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)  # H2O предпочитает float32/64

class Model():
    def __init__(self, model_path: str,
                 name_df: str,
                 output_feature: str):

        self.model_path = model_path
        self.name_df = name_df
        self.output_feature = output_feature

    def load_model(self):
        model = import_mojo(self.model_path)

        return model

    def prediction(self, input_data):
        h2o.init()

        input_data = pd.DataFrame({
            "smile": [input_data]
        })

        fingerprints = np.array([smiles_to_morgan_fp(s) for s in input_data["smile"]])
        feature_names = [f"fp_{i}" for i in range(fingerprints.shape[1])]
        X_df = pd.DataFrame(fingerprints, columns=feature_names)

        input_data = h2o.H2OFrame(X_df)
        print(self.model_path)
        model = h2o.import_mojo(self.model_path)
        predictions = model.predict(input_data)

        return predictions


