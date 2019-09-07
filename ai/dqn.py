#!/usr/bin/python3

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
# from keras.callbacks import TensorBoard
# from collections import deque
import time
import numpy as np
import tensorflow as tf
import random
from collections import deque
import pickle
import itertools

from game import Game

MEMORY_FILE = 'output/dqn/memory.bin'
MODEL_FILE = 'output/dqn/model.h5'

random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

class Dqn:
  def __init__(self, game, memory_file = MEMORY_FILE, **kwargs):
    self.game = game
    self.memory_file = memory_file

    try:
      with open(self.memory_file, 'rb') as file:
        self.memory = pickle.load(file)
    except Exception as e:
      print(f"Could not load memory. ex: {e}\nInitializing empty memory.")
      self.memory = deque(maxlen = 25_000)
    self.gamma = 0.95 # discount rate
    self.epsilon = 0.95 # exploration rate
    self.epsilon_min = 0.25
    self.epsilon_decay = 0.998
    self.learning_rate = 0.002
    try:
      self.model = load_model(MODEL_FILE)
      # raise Exception() # debug
    except Exception as e:
      print(f"Could not load model. ex: {e}\nInitializing new model.")
      state = self.image_to_state(self.game.step(None)[0])
      self.model = self.create_model(state.shape)
    finally:
      print(f"Input: {self.model.input_shape}")
      self.model.summary()
  
  def close(self):
    with open(self.memory_file, 'wb') as file:
      pickle.dump(self.memory, file)
    self.model.save(MODEL_FILE)
  
  def image_to_state(self, image):
    im = image # .convert("L")
    # im.show()
    arr = np.array(img_to_array(im)) / 255
    return arr

  def create_model(self, input_shape):
    model = Sequential()

    model.add(Conv2D(8, (9, 9), strides=(3, 3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(16, (7, 7), strides=(2, 2), activation="relu"))
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (5, 5), activation="relu"))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Dense(len(self.game.actions), activation="linear"))

    model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])

    return model
  
  def act(self, state):
    if random.random() < self.epsilon or state is None:
      return random.randrange(len(self.game.actions)), None
    else:
      prediction = self.predict(state)
      action = max(enumerate(prediction), key=lambda e: e[1])
      print(f"predicting reward of {action[1]}")
      return action[0], prediction
  
  def predict(self, state):
    return self.model.predict(state.reshape(-1, *state.shape))[0]

  def remember(self, state, action, reward, next_state, done, reward_delta):
    self.memory.append((state, action, reward, next_state, done, reward_delta))
  
  def replay(self, batch_size):
    len_memory = len(self.memory)
    if len_memory < batch_size:
      return False
    self.train(batch_size, 1, list(itertools.islice(self.memory, len_memory - batch_size, len_memory)))
    print(f"Memory: {len(self.memory)}")
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
  
  def train(self, batch_size, epochs = 1, memory = None):
    if memory is None:
      memory = self.memory
    if len(memory) < batch_size:
      return False

    current_qs = self.model.predict(np.array([e[0] for e in memory]))
    future_qs = self.model.predict(np.array([e[3] for e in memory]))

    x = []
    y = []

    for index, (state, action, reward, next_state, done, reward_delta) in enumerate(memory):
      q = reward_delta
      if not done:
        reward += self.gamma * np.amax(future_qs[index])
      
      qs = current_qs[index]
      qs[action] = q

      x.append(state)
      y.append(qs)
    
    self.model.fit(np.array(x), np.array(y), batch_size = batch_size, shuffle = True, epochs = epochs)
