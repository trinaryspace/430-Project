import pyautogui
import time


'''
#print("Quick! Move your mouse over the 'Record' button...")

# 5-second countdown
for i in range(5, 0, -1):
    print(f"Reading in {i} seconds...")
    time.sleep(1)

# Get and print the position
x, y = pyautogui.position()
print(f"\nDone! The coordinates are: X: {x}, Y: {y}")

#coords on my 4k laptop monitor: 2383,132
#coords on laptop plus monitor at home: 2142, -346
pyautogui.FAILSAFE = True # move to top left corner of screen to break the program

'''

# Give yourself a few seconds to focus the DAQ window before it starts
time.sleep(3) 

while True:
    # Replace with the actual coordinates of the record button
    pyautogui.click(x=2142, y=-346) 
    time.sleep(0.3) # 2 Hz click rate
    pyautogui.press('enter')
    time.sleep(0.15)



