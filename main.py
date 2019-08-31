#!/usr/bin/python3

import sys
import pyscreenshot as ImageGrab
import time
import pynput.mouse
from pynput import keyboard

from ai.dqn import Dqn as Ai

BLACK = (0, 0, 0)

paused = False

def main():
  if len(sys.argv) < (1 + 4 + 2 + 1):
    print(f"Usage: {sys.argv[0]} <x1> <y1> <x2> <y2> <cx> <cy> <inv_scale>") # 0 300 956 840 20 20 2
    exit(1)

  application_bbox = tuple(int(i) for i in sys.argv[1 : 5])
  application_height = application_bbox[3] - application_bbox[1]
  application_width = application_bbox[2] - application_bbox[0]
  application_scale = float(sys.argv[7])
  scaled_application_size = (int(application_width / application_scale), int(application_height / application_scale))
  print(f"Scaled application size: {scaled_application_size}")
  stats_bbox = (
    application_width / 24.5128,
    application_height / 11.4894,
    application_width / 9.56,
    application_height / 6.352941176) # 32 33 170 100
  previous_image = ImageGrab.grab(bbox=application_bbox)
  previous_stats = previous_image.crop(box=stats_bbox)
  previous_stats_pixels = previous_stats.load()

  # previous_image.show()
  print("If this screenshot did not capture the game, restart the application with adjusted arguments.")

  mouse = pynput.mouse.Controller()

  actions_x = int((int(sys.argv[3]) - int(sys.argv[1])) / int(sys.argv[5]))
  actions_y = int((int(sys.argv[4]) - int(sys.argv[2])) / int(sys.argv[6]))
  actions_count = actions_x * actions_y
  ai = Ai(actions_count=actions_count)

  with keyboard.Listener(
    on_press = on_press
  ) as listener:
    while True:
      if paused:
        print(".", end="", flush=True)
        time.sleep(0.5)
        continue
      image = ImageGrab.grab(bbox=application_bbox)
      stats = image.crop(box=stats_bbox)
      stats_pixels = stats.load()

      scaled_image = image.resize(scaled_application_size)
      # scaled_image.show()

      action = ai.process(image)
      print(action, flush=True)

      game_over = False
      for x in range(stats.size[0]):
        if stats_pixels[x, stats.size[1] - 1] == BLACK:
          game_over = True
          break
      if game_over:
        ai.game_over()
        break # TODO: restart?
      else:
        next_level = False
        for x in range(stats.size[0] - 1):
          for y in range(stats.size[1] - 1):
            if stats_pixels[x, y] != previous_stats_pixels[x, y]:
              next_level = True
              break
          if next_level:
            break
        if next_level:
          ai.next_level()

      previous_image = image
      previous_stats = stats
      previous_stats_pixels = stats_pixels

      time.sleep(1.000)
      # break

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