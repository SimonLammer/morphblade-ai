#!/usr/bin/python3

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
# from keras.callbacks import TensorBoard
# from collections import deque
import time
import numpy as np
import tensorflow as tf
import random

from game import Game

random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

class Dqn:
  def __init__(self, game, **kwargs):
    self.game = game

    state = self.image_to_state(self.game.step(None)[0])
    self.model = self.create_model(state.shape)
  
  def image_to_state(self, image):
    return np.array(img_to_array(image))

  def create_model(self, input_shape):
    model = Sequential()

    model.add(Conv2D(64, (7, 7), input_shape=input_shape, activation="relu"))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(len(self.game.actions), activation="softmax"))
    model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

    return model
