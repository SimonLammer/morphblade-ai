#!/usr/bin/python3

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.callbacks import TensorBoard
from collections import deque
import time
import numpy as np
import tensorflow as tf

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MODEL_NAME = "256x2"

class Dqn:
  def __init__(self, **kwargs):
    self.model = None
    self.config = kwargs
 
  def _lateinit(self, image_array):
    self.model = self.create_model(image_array.shape) # gets trained
    self.target_model = self.create_model(image_array.shape) # used for .predict
    self.target_model.set_weights(self.model.get_weights())
    self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
    self.target_update_counter = 0

  def process(self, image):
    arr = img_to_array(image)
    if not self.model:
      self._lateinit(arr)
    return None
 
  def game_over(self):
    print("GAME OVER")

  def next_level(self):
    print("NEXT LEVEL")

  def create_model(self, input_shape):
    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(self.config['actions_count'], activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

    return model

  def updaet_replay_memory(self, transition):
    self.replay_memory.append(transition)

  def get_qs(self):
    return self.model_predict(np.array(state).reshape(-1, *state.shape) / 255)[0]



class ModifiedTensorBoard(TensorBoard):
  """
  https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
  """

  # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.step = 1
    self.writer = tf.summary.FileWriter(self.log_dir)

  # Overriding this method to stop creating default log writer
  def set_model(self, model):
    pass

  # Overrided, saves logs with our step number
  # (otherwise every .fit() will start writing from 0th step)
  def on_epoch_end(self, epoch, logs=None):
    self.update_stats(**logs)

  # Overrided
  # We train for one batch only, no need to save anything at epoch end
  def on_batch_end(self, batch, logs=None):
    pass

  # Overrided, so won't close writer
  def on_train_end(self, _):
    pass

  # Custom method for saving own metrics
  # Creates writer, writes custom metrics and closes writer
  def update_stats(self, **stats):
    self._write_logs(stats, self.step)
