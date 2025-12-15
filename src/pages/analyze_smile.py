from utils.model_prediction import Model
from utils.model_sav_prediction import predict_sav
from utils.analysis_generator import llm_question
from utils.output_info import transform_info
from utils.formula_mol import smiles_to_formula

import asyncio
import os
from io import BytesIO

import streamlit as st
from rdkit.Chem import Draw
from rdkit import Chem

path_models = "src/models"

model_and_outputs = {
    "HIV": {
        "output_feature": "HIV_active"
    },
    "BBBP": {
        "output_feature": "p_np"
    },
    "base": {
            "output_feature": ["pIC50", "Class"]
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
    global path_models
    global model_and_outputs
    predictions = {}

    for name_df in list_df:
        feature_output = model_and_outputs.get(name_df).get("output_feature")

        if name_df in list(model_and_outputs.keys()):
            name_file = os.listdir(path_models + "/" + name_df)[0]
            model_path = path_models + "/" + name_df + "/" + name_file

            if (name_df != "base" and name_df != "FreeSolkSAMPL"):
                model = Model(
                    model_path,
                    name_df,
                    feature_output
                )

                predict = model.prediction(input_smile)
                # st.write(predict)
                predictions.update(
                    {
                        name_df: {
                            "predict": predict,
                            "output_feature": feature_output
                        }
                    }
                )
            elif (name_df == "FreeSolkSAMPL"):
                predict = await predict_sav(
                    input_smile,
                    model_path
                )
                #
                # st.write(model_path)
                # st.write(predict)

                predictions.update(
                    {
                        name_df: {
                            "predict": predict[0],
                            "output_feature": feature_output
                        }
                    }
                )

            elif (name_df == "base"):
                features = {}
                name_df = path_models + "/" + name_df

                for name_file in os.listdir(name_df):
                    model_path = name_df + "/" + name_file

                    predict, X_df = await predict_sav(
                        input_smile,
                        model_path
                    )
                    st.session_state.X_df = X_df

                    features.update(
                        {
                            name_file.split(".")[0]: predict
                        }
                    )

                predictions.update(
                    {
                        name_df: {
                            "predict": features
                        }
                    }
                )

    return predictions

async def main():
    global path_models

    smile = st.session_state.get("smile")
    molekul_chem = Chem.MolFromSmiles(smile)

    img_molekul = Draw.MolToImage(molekul_chem, size=(900, 600))
    buffer = BytesIO()
    img_molekul.save(buffer, format="PNG")
    buffer.seek(0)

    st.image(
        buffer,
        caption="Molecule from SMILES"
    )

    formula = smiles_to_formula(smile)
    st.write(f"Химическая формула молекулы: {formula}")


    list_df = os.listdir(path_models)
    predictions = await all_predictions(list_df,
                                        smile)

    await transform_info(predictions)

    response_model = await llm_question(st.session_state.get("X_df"),
                                        predictions,
                                        smile)

    st.subheader("Дополнительный анализ модели параметров")
    st.write(response_model)

try:
    asyncio.run(main())
except:
    st.info("В вводе допущена ошибка")

