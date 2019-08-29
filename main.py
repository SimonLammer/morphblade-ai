#!/usr/bin/python3

import sys
import pyscreenshot as ImageGrab
import time
import pynput.mouse
from pynput import keyboard

from ai.blank import Blank as Ai

paused = True

def main():
  if len(sys.argv) < 5:
    print(f"Usage: {sys.argv[0]} <x1> <y1> <x2> <y2>{sys.argv[0]}") # 0 300 956 840
    exit(1)

  application_bbox = tuple(int(i) for i in sys.argv[1 : 5])
  image = ImageGrab.grab(bbox=application_bbox)

  # image.show()
  print("If this screenshot did not capture the game, restart the application with adjusted arguments.")

  mouse = pynput.mouse.Controller()

  ai = Ai()

  with keyboard.Listener(
    on_press = on_press
  ) as listener:
    while True:
      if paused:
        print(".", end="", flush=True)
        time.sleep(0.5)
        continue

      action = ai.process(image)
      print(action, flush=True)
      time.sleep(1.000)

    listener.join()

def on_press(key):
  global paused
  # import pdb; pdb.set_trace()
  if hasattr(key, 'char'):
    if key.char == 'i':
      paused = False
      print("Started AI (press 'o' to stop)")
    elif key.char == 'o':
      paused = True
      print("Stopped AI")

if __name__ == '__main__':
  main()