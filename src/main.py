import asyncio
from pyexpat import native_encoding

import streamlit as st

async def main():
    variants = ["Свой дрон", "Возьму в аренду"]
    type_bpla = st.radio(
        "Выберите тип дрона",
        variants,
        index=None,
        key="choice"
    )

    goal = st.text_input("Введите цель полёта")

    all_components = [type_bpla]

    if st.button("Далее"):
        for element in all_components:
            if not element:
                st.info("Вы заполнили не все компоненты")
                break
        else:
            if type_bpla == "Свой дрон":
                st.switch_page("pages/your_drone.py")
            elif type_bpla == "Возьму в аренду":
                st.switch_page("pages/rent_drone.py")

if __name__ == "__main__":
    asyncio.run(main())