# pip install pyautogui
import pyautogui

def func(data):
    if isinstance(data, str) and len(data) == 1 and data.isalpha():
        # 如果輸入是字元，讓鍵盤輸入對應的字元
        pyautogui.press(data)
    else:
        # 如果是字元以外，就讓滑鼠移動一點
        current_position = pyautogui.position()
        pyautogui.moveTo(current_position[0] + 100, current_position[1])
        
        # 如果是字元以外，就滑鼠點擊左鍵一下
        # pyautogui.click()

        # 如果是字元以外，就滑鼠持續往右移
        # pyautogui.move(10, 0)

# 測試函數
input_char = input(4)
func(input_char)