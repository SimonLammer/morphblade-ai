#!/usr/bin/python3

import sys
from datetime import datetime
import time
import random
from pynput import keyboard
import numpy as np
from PIL import ImageDraw

from ai.dqn import Dqn as Ai
from game import Game

MIN_PREDICTION_SIZE = 2
MAX_PREDICTION_SIZE = 15
PREDICTION_COLOR = "magenta"

keep_running = True

def main():
  if len(sys.argv) < (1 + 4 + 1):
    print(f"Usage: {sys.argv[0]} <x1> <y1> <x2> <y2> <scale>") # 0 300 956 840 1
    exit(1)

  application_bbox = tuple(int(i) for i in sys.argv[1 : 5])
  board_scale = float(sys.argv[5])
  game = Game(application_bbox, board_scale)
  # game.demo_capture()
  print("If this screenshot did not capture the game, restart the application with adjusted arguments.")
  # time.sleep(2)

  ai = Ai(game)

  # exit(0)

  time.sleep(1) # Give user time to switch to application

  listener = keyboard.Listener(on_press = on_press)
  listener.start()
  state = None
  while keep_running:
    action_index, prediction = ai.act(state)
    if prediction is not None:
      save(game, board, prediction)

    scaled_board, reward, done, board = game.step(action_index)
    next_state = ai.image_to_state(scaled_board)

    # x = next_state.reshape(-1, *next_state.shape)
    # y = np.array([1] + [0] * 36).reshape(1, -1)
    # ai.model.fit(x, y, epochs=1)
    # print(ai.model.predict(next_state.reshape(-1, *next_state.shape)))

    if state is not None:
      ai.remember(state, action_index, reward, next_state, done)
      print(reward, done)

    if done:
      print("Restarting")
      game.restart()
      ai.replay(32)
      print(f"epsilon: {ai.epsilon}")
    # time.sleep(3) # debug
    state = next_state
  ai.close()
  
  listener.stop()

def save(game, board, prediction):
  im = board.copy()
  draw = ImageDraw.Draw(im)
  for i in range(len(game.actions)):
    confidence = prediction[i]
    action = list(game.actions[i])
    for j in range(2):
      action[j] -= game.board_bbox[j]

    size = confidence * (MAX_PREDICTION_SIZE - MIN_PREDICTION_SIZE) + MIN_PREDICTION_SIZE
    width = int(size / 5)
    draw.line((action[0] + size, action[1] + size, action[0] - size, action[1] - size), fill = PREDICTION_COLOR)
    draw.line((action[0] - size, action[1] + size, action[0] + size, action[1] - size), fill = PREDICTION_COLOR, width = width)

  im.save(f'output/predictions/{datetime.utcnow().strftime("%Y%m%dT%H%M%S")}.jpg')

def on_press(key):
  global keep_running
  if key == keyboard.Key.esc:
    print("Halting due to ESC press", file=sys.stdout)
    keep_running = False
    return False

if __name__ == '__main__':
  main()