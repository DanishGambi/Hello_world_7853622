import asyncio

import streamlit as st

async def main():
    st.title("Заполнение документов для заказа дрона")
    name_dron = st.text_input("Введите название дрона который вы выбрали")

    types_zyav = ["ИП", "Физ. лицо", "Юр. лицо"]
    type_zyav = st.radio(
        "Выберите тип заявителя",
        types_zyav,
        index=None
    )

    goal = st.text_input("Введите цель полёта")

    class_dron = ["класс G уведомительный порядок", "Класс A/C Разрешительный порядок"]
    class_bpla = st.radio(
        "Выберите тип заявителя",
        class_dron,
        index=None
    )

    all_components = [name_dron, types_zyav, class_dron]

    if st.button("Далее"):
        for element in all_components:
            if not element:
                st.info("Вы заполнили не все компоненты")
                break
        else:
            st.switch_page("pages/map.py")

asyncio.run(main())