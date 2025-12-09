import asyncio

import streamlit as st
from streamlit_folium import st_folium
import folium

async def main():
    # Инициализация списка выбранных точек в session_state
    st.write("Выберите координаты точек")

    if "selected_points" not in st.session_state:
        st.session_state.selected_points = []

    # Центрируем карту (можно изменить на нужные координаты)
    initial_lat = 55.99886
    initial_lon = 92.752259  # Красноярск

    # Создаём карту Folium
    m = folium.Map(
        location=[initial_lat, initial_lon],
        zoom_start=10,
        tiles="OpenStreetMap"
    )

    # Добавляем уже выбранные точки (если есть)
    for point in st.session_state.selected_points:
        folium.Marker(
            location=[point["lat"], point["lng"]],
            popup=f"Lat: {point['lat']:.4f}, Lng: {point['lng']:.4f}",
            icon=folium.Icon(color="red")
        ).add_to(m)

    # Отображаем карту и ловим клики
    output = st_folium(
        m,
        width=700,
        height=500,
        key="map"
    )

    # Обработка клика по карте
    if output["last_clicked"] is not None:
        lat = output["last_clicked"]["lat"]
        lng = output["last_clicked"]["lng"]

        # Добавляем точку, если её ещё нет (опционально: избегаем дублей)
        new_point = {"lat": lat, "lng": lng}
        if not any(p["lat"] == lat and p["lng"] == lng for p in st.session_state.selected_points):
            st.session_state.selected_points.append(new_point)
            st.rerun()  # Обновляем страницу, чтобы обновить карту с новой точкой

    # Отображение списка координат
    st.subheader("Выбранные координаты:")
    if st.session_state.selected_points:
        for i, point in enumerate(st.session_state.selected_points):
            st.write(f"{i + 1}. Широта: {point['lat']:.6f}, Долгота: {point['lng']:.6f}")

        # Кнопка для очистки
        if st.button("Очистить все точки"):
            st.session_state.selected_points.clear()
            st.rerun()
    else:
        st.write("Кликните по карте, чтобы добавить точки.")
    #
asyncio.run(main())