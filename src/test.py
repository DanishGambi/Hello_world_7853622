import pyautogui
import time
import webbrowser

# Открыть сайт
webbrowser.open("https://sppi.ivprf.ru/check-uav")
time.sleep(5)  # Ждём загрузки

# ⚠️ Теперь нужно вручную пройти авторизацию через Госуслуги!
# Или автоматизировать ввод (НЕ РЕКОМЕНДУЕТСЯ):
# pyautogui.click(x=..., y=...)  # клик по "Войти через Госуслуги"
# pyautogui.typewrite("ваш_логин")
# pyautogui.press("tab")
# pyautogui.typewrite("ваш_пароль")
# pyautogui.press("enter")

# После входа — заполнение формы (примерные координаты!)
# time.sleep(10)  # Ждём входа
DJI Mavic 3
# Пример: клик в поле "Модель дрона"
pyautogui.click(x=500, y=300)
pyautogui.typewrite("DJI Mavic 3")

# Поле "Цель полёта"
pyautogui.click(x=500, y=350)
pyautogui.typewrite("Аэрофотосъёмка")

# Нажать кнопку "Проверить"
pyautogui.click(x=600, y=500)

time.sleep(3)

# Сделать скриншот результата
screenshot = pyautogui.screenshot()
screenshot.save("result.png")
print("Скриншот результата сохранён как 'result.png'")