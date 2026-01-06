"""
Modify the shortcuts for the gestures.

Please read the PyAutoGUI documentation at https://pyautogui.readthedocs.io/en/latest/.
"""

import pyautogui


def left_gesture():
    pyautogui.hotkey('alt', 'left')


def right_gesture():
    pyautogui.hotkey('alt', 'right')


def up_gesture():
    pass
    # pyautogui.hotkey('win', 'd')


def down_gesture():
    pass
    # pyautogui.hotkey('ctrl', 'r')
