#!/usr/bin/python3

import time
from sys import stderr
import pyscreenshot as ImageGrab
from PIL import ImageDraw
from pynput.mouse import Button as MouseButton, Controller as MouseController

ACTION_SIZE = 4
ACTION_OUTLINE = "magenta"
BOARD_OUTLINE = "cyan"
STATS_OUTLINE = "orange"

DEFAULT_REWARD = 0
GAME_OVER_REWARD = -1
LEVEL_COMPLETE_REWARD = 1


BLACK = (0, 0, 0)

ACTION_UPDATE_TIME = 0.6
CLICK_TIME = 0.025
MENU_CLICK_TIME = 0.025
MENU_LOAD_TIME = 0.250
MENU_POST_TIME = 0.5

MOUSE = MouseController()

class Game:
  def __init__(self, application_bbox, board_scale = 1):
    self.application_bbox = application_bbox
    self.application_width = self.application_bbox[2] - self.application_bbox[0]
    self.application_height = self.application_bbox[3] - self.application_bbox[1]
    self.board_scale = board_scale
    self.board_bbox = (
      int(self.application_width / 3.9833),
      int(self.application_height / 9.8182),
      int(self.application_width / 1.32778),
      int(self.application_height / 1.1134))
    self.board_width = self.board_bbox[2] - self.board_bbox[0]
    self.board_height = self.board_bbox[3] - self.board_bbox[1]
    self.stats_bbox = (
      int(self.application_width / 24.5128),
      int(self.application_height / 11.4894),
      int(self.application_width / 9.56),
      int(self.application_height / 6.35294))
    self.stats_width = self.stats_bbox[2] - self.stats_bbox[0]
    self.stats_height = self.stats_bbox[3] - self.stats_bbox[1]
    self.menu_coordinates = (
      int(self.application_bbox[0] + self.application_width / 18.38461),
      int(self.application_bbox[1] + self.application_height / 9.15254))
    self.menu_restart_coordinates = (
      int(self.application_bbox[0] + self.application_width / 2.1292),
      int(self.application_bbox[1] + self.application_height / 2.44348))
    self._stats = None
    self._reward = 0

    self._generate_actions()
    self.step(None)
  
  def _generate_actions(self):
    self.actions = []
    # Middle: 480, 452
    # lu: 388, 111
    down_left = (
      -self.application_width / 31.86667,
      self.application_height / 10.3846)
    right = (
      self.application_width / 15.67213,
      0)
    field = [ # top left
      self.application_width / 2.46392,
      self.application_height / 4.86487]
    for diagonal_lengths, shrink in [
      (range(4, 8), False),
      (range(6, 3, -1), True)
      ]:
      for diagonal_length in diagonal_lengths:
        if shrink:
          for i in range(2):
            field[i] += down_left[i]
        f = field.copy()
        for row in range(diagonal_length):
          self.actions.append((int(f[0]), int(f[1])))
          for i in range(2):
            f[i] += down_left[i]
        field[0] += right[0]

  def demo_capture(self):
    application = ImageGrab.grab(bbox=self.application_bbox)
    draw = ImageDraw.Draw(application)
    draw.rectangle(self.stats_bbox, outline = STATS_OUTLINE)
    draw.rectangle(self.board_bbox, outline = BOARD_OUTLINE)
    for x, y in self.actions:
      draw.line((x - ACTION_SIZE, y - ACTION_SIZE, x + ACTION_SIZE, y + ACTION_SIZE), fill = ACTION_OUTLINE)
      draw.line((x + ACTION_SIZE, y - ACTION_SIZE, x - ACTION_SIZE, y + ACTION_SIZE), fill = ACTION_OUTLINE)
    application.show()

  def step(self, action_index):
    done = False
    reward_delta = DEFAULT_REWARD

    if self._stats is not None:
      if action_index is not None:
        action = self.actions[action_index]
        MOUSE.position = [action[i] + self.application_bbox[i] for i in range(2)]
        MOUSE.press(MouseButton.left)
        time.sleep(CLICK_TIME)
        MOUSE.release(MouseButton.left)

        time.sleep(ACTION_UPDATE_TIME)

    application = ImageGrab.grab(bbox=self.application_bbox)
    stats = application.crop(box=self.stats_bbox).load()

    if self._stats is not None:
      if self._game_over(stats):
        done = True
        reward_delta = GAME_OVER_REWARD
      else:
        next_level = False
        for x in range(self.stats_width - 1):
          for y in range(self.stats_height - 1):
            if stats[x, y] != self._stats[x, y]:
              next_level = True
              break
          if next_level:
            break
        if next_level:
          reward_delta = LEVEL_COMPLETE_REWARD
    
    self._stats = stats
    self._reward += reward_delta
    new_state = application.crop(box=self.board_bbox)
    scaled_new_state = new_state.resize((
      int(self.board_width * self.board_scale),
      int(self.board_height * self.board_scale)))
    return scaled_new_state, self._reward, done, new_state, reward_delta

  def capture_screen(self):
    return ImageGrab.grab(bbox=application_bbox)
    scaled_screen = image.resize(scaled_application_size)
    return scaled_screen

  def _game_over(self, stats):
    game_over = False
    for x in range(self.stats_width):
      if stats[x, self.stats_height - 1] == BLACK:
        game_over = True
        break
    return game_over
  
  def restart(self):
    self._reward = 0
    cnt = 0
    while True:
      MOUSE.position = self.menu_coordinates
      MOUSE.press(MouseButton.left)
      time.sleep(MENU_CLICK_TIME)
      MOUSE.release(MouseButton.left)
      time.sleep(MENU_LOAD_TIME)
      MOUSE.position = self.menu_restart_coordinates
      MOUSE.press(MouseButton.left)
      time.sleep(MENU_CLICK_TIME)
      MOUSE.release(MouseButton.left)

      time.sleep(MENU_POST_TIME)

      stats = ImageGrab.grab(bbox=self.application_bbox).crop(box=self.stats_bbox).load()
      if not self._game_over(stats):
        self._stats = stats
        break
      print(cnt)
      if cnt > 3:
        print("Could not restart game", file=stderr)
        exit(1)
      cnt += 1
