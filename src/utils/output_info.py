import streamlit as st

all_molekul = {
    "BBBP": " бинарные метки проникновения гематоэнцефалического барьера (проницаемость)(p_np).",
    "ClinTox": "качественные данные о препаратах, одобренных FDA, и тех, "
               "которые не прошли клинические испытания по причинам токсичности (FDA_APPROVED)",
    "ESOL": "данные о растворимости в воде (логарифмическая растворимость в молях на литр) "
            "для распространённых органических малых молекул (ESOL predicted log solubility in mols per litre).",
    "HIV": "экспериментально измеряемые способности ингибировать репликацию ВИЧ (HIV_active).",
    "Lipophilicity": " экспериментальные результаты коэффициента распределения октанола/воды (logD при pH 7,4) (exp)",
    "bace": ["Количественные результаты связывания для набора ингибиторов человеческой β-секретазы 1 (pIC50)",
             "качественные (бинарные метки) результаты связывания для набора ингибиторов человеческой β-секретазы 1. Обозначает активность/инактивность ингибитора (например, 0 для не-связывающегося, 1 для связывающегося) (Class)."],
    "FreeSolkSAMPL": "рассчитанная энергия без гидратации малых молекул в воде (calc)."
}

async def transform_info(predicts: dict):
    st.subheader("Результаты свойств предсказания модели")

    for name_df in predicts.keys():
        if name_df == "base":
            description = all_molekul.get("bace")
            pred = predicts.get(name_df).get("predict")

            for desc, key  in zip(description, list(pred.keys())):
                st.write(f"{name_df}   {desc}")
                st.info(f"Значение параметра: {pred.get(key)}")
        else:
            description = all_molekul.get(name_df)
            pred = predicts.get(name_df).get("predict")
            st.write(f"{name_df}   {description}")
            st.info(f"Значение параметра: {pred}")