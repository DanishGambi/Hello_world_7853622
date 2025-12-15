import asyncio
import os
from io import BytesIO

import streamlit as st
from rdkit import Chem

async def main():
    st.title("Анализ свойств молекулы")

    smile = st.text_input("Введите молекулу")

    # uploaded_file = st.file_uploader(
    #     "Загрузите изображение",
    #     type=["csv"],
    #     accept_multiple_files=False,
    #     help="Поддерживаются форматы: CSV"
    # )

    button_continue = st.button("Далее")

    if (button_continue):
        if not (smile):
            st.info("Вы заполнили не все компоненты")
        else:
            if smile:
                st.session_state.smile = smile

                st.switch_page(
                    "pages/analyze_smile.py"
                )
            else:
                pass

if __name__ == "__main__":
    asyncio.run(main())