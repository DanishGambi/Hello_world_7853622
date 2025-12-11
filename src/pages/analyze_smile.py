from utils.model_prediction import Model

import asyncio
import os
from io import BytesIO

import streamlit as st
from rdkit.Chem import Draw
from rdkit import Chem

model_and_outputs = {
    "HIV": {
        "output_feature": "HIV_active"
    },
    "BBBP": {
        "output_feature": "p_np"
    },
    "base": {
            "output_feature": "pIC50"
        },
    "FreeSolkSAMPL": {
        "output_feature": "expt"
    },
    "ESOL": {
        "output_feature": "ESOL predicted log solubility in mols per litre"
    },
    "Lipophilicity": {
        "output_feature": "exp"
    },
    "ClinTox": {
        "output_feature": "CT_TOX"
    }
}


async def all_predictions(list_df: list[str],
                          input_smile):
    global model_and_outputs
    predictions = {}

    for name_df in list_df:
        st.write(model_and_outputs.get(name_df))
        feature_output = model_and_outputs.get(name_df).get("output_feature")

        if name_df in list(model_and_outputs.keys()):
            model_path = name_df + "/" + os.listdir(name_df)[0]
            st.write(model_path)
            model = Model(
                model_path,
                name_df,
                feature_output
            )

            predict = model.prediction(input_smile)
            st.write(predict)
            predictions.update(
                {
                    name_df: {
                        "predict": predict.as_data_frame(),
                        "output_feature": feature_output
                    }
                }
            )

    return predictions

async def main():
    smile = st.session_state.get("smile")
    molekul_chem = Chem.MolFromSmiles(smile)

    img_molekul = Draw.MolToImage(molekul_chem, size=(400, 400))
    buffer = BytesIO()
    img_molekul.save(buffer, format="PNG")
    buffer.seek(0)

    st.write(os.getcwd())
    os.chdir("models")
    list_df = os.listdir()
    st.write(list_df)
    predictions = await all_predictions(list_df,
                                        smile)
    st.write(predictions)

    st.image(
        buffer,
        caption = "Molecule from SMILES"
    )

asyncio.run(main())