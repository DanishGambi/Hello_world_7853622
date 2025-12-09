import asyncio

import streamlit as st


async def main():
    st.title("Выбор точки маршрута дрона")

    drone_names = [
        "SkyHawk",
        "AeroVista",
        "Nimbus X1",
        "PhantomEye",
        "Orion Drone",
        "VantaFly",
        "Zenith Scout",
        "Echelon Air",
        "NovaWing",
        "PulseDrone"
    ]

    class_bpla = st.radio(
        "Выберите тип заявителя",
        drone_names,
        index=None
        )

    all_components = [class_bpla]

    if st.button("Далее"):
        for element in all_components:
            if not element:
                st.info("Выберите доступный беспилотник")
                break
        else:
            st.switch_page("pages/map.py")

asyncio.run(main())