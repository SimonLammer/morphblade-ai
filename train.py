#!/usr/bin/python3

import sys
from collections import deque
import random

from game import Game
from ai.dqn import Dqn as Ai

def main():
  if len(sys.argv) < (1 + 4 + 1 + 1):
    print(f"Usage: {sys.argv[0]} <x1> <y1> <x2> <y2> <scale> <memory_file>")
    exit(1)

  application_bbox = tuple(int(i) for i in sys.argv[1 : 5])
  board_scale = float(sys.argv[5])
  memory_file = sys.argv[6]

  game = Game(application_bbox, board_scale)
  ai = Ai(game, memory_file)

  print(f"Memory size: {len(ai.memory)}")

  ai.train(128, 16)

  ai.close()


if __name__ == '__main__':
  main()
