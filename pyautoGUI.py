import pyautogui

def func(char):
    if len(char) == 1 and char.isalpha():
        pyautogui.press(char)
    else:
        current_position = pyautogui.position()
        pyautogui.moveTo(current_position[0] + 100, current_position[1])

input_char = input("Please enter characters or other content: ")
func(input_char)
