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

from game import Game

MEMORY_FILE = 'output/dqn/memory.bin'
MODEL_FILE = 'output/dqn/model.h5'

random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

class Dqn:
  def __init__(self, game, **kwargs):
    self.game = game

    try:
      with open(MEMORY_FILE, 'rb') as file:
        self.memory = pickle.load(file)
    except Exception as e:
      print(f"Could not load memory. ex: {e}\nInitializing empty memory.")
      self.memory = deque(maxlen = 2000)
    self.gamma = 0.95 # discount rate
    self.epsilon = 0.9 # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.learning_rate = 0.001

    state = self.image_to_state(self.game.step(None)[0])
    try:
      self.model = load_model(MODEL_FILE)
      raise Exception() # debug
    except Exception as e:
      print(f"Could not load model. ex: {e}\nInitializing new model.")
      self.model = self.create_model(state.shape)
    finally:
      print(f"Input: {self.model.input_shape}")
      self.model.summary()
  
  def close(self):
    with open(MEMORY_FILE, 'wb') as file:
      pickle.dump(self.memory, file)
    self.model.save(MODEL_FILE)
  
  def image_to_state(self, image):
    im = image.convert("L")
    # im.show()
    arr = np.array(img_to_array(im)) / 255
    return arr

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
    model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])

    return model
  
  def act(self, state):
    if random.random() < self.epsilon or state is None:
      return random.randrange(len(self.game.actions)), None
    else:
      prediction = self.predict(state)
      action_index = max(enumerate(prediction), key=lambda e: e[1])[0]
      return action_index, prediction
  
  def predict(self, state):
    return self.model.predict(state.reshape(-1, *state.shape))[0]

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
  
  def replay(self, batch_size):
    if len(self.memory) < batch_size:
      return
    minibatch = random.sample(self.memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
      target = reward
      if not done:
        target += self.gamma * np.amax(self.predict(next_state))
      target_f = np.array(self.predict(state))
      target_f[action] = target
      self.model.fit(state.reshape(-1, *state.shape), target_f.reshape(1, -1), epochs = 1, verbose = 1)
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
