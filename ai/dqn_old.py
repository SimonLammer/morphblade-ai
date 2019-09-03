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
import random

DISCOUNT = 0.99
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
MODEL_NAME = "256x2"
REPLAY_MEMORY_SIZE = 50_000
UPDATE_TARGET_EVERY = 5

EPISODES = 20

epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 50
SHOW_PREVIEW = False

MIN_REWARD = -100

random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

class Dqn:
  def __init__(self, **kwargs):
    self.action = 0
    self.done = False
    self.model = None
    self.config = kwargs
    self.previous_state = None
    self.step = 0
    self.reward = 0
    self.ep_rewards = [MIN_REWARD]
 
  def _lateinit(self, image_array):
    self.model = self.create_model(image_array.shape) # gets trained
    self.target_model = self.create_model(image_array.shape) # used for .predict
    self.target_model.set_weights(self.model.get_weights())
    self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
    self.target_update_counter = 0

  def process(self, image):
    state = img_to_array(image)
    if not self.model:
      self._lateinit(state)

    if self.previous_state is not None:
      self.update_replay_memory((self.previous_state, self.action, self.reward, state, self.done))
      self.train(self.done, self.step)

      # self.ep_rewards.append(episode_reward)
      # if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        # average_reward = sum(self.ep_rewards[-AGGREGATE_STATS_EVERY:])/len(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
        # min_reward = min(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
        # max_reward = max(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
        # self.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        # if min_reward >= MIN_REWARD:
            # self.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')


    if np.random.random() > epsilon:
      self.action = np.argmax(self.get_qs(state))
    else:
      self.action = np.random.randint(0, self.config['actions_count'])

    self.step += 1
    reward = -1
    self.previous_state = state
    return None
 
  def game_over(self):
    # self.done = True
    self.reward = -100

  def next_level(self):
    self.reward = 100

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

  def update_replay_memory(self, transition):
    self.replay_memory.append(transition)

  def get_qs(self, state):
    return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

  def train(self, terminal_state, step):
    if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
      return
    
    minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

    current_states = np.array([transition[0] for transition in minibatch]) / 255
    current_qs_list = self.model.predict(current_states)

    new_current_states = np.array([transition[3] for transition in minibatch]) / 255
    future_qs_list = self.target_model.predict(new_current_states)

    x = []
    y = []
    
    for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
      if done:
        new_q = reward
      else:
        max_future_q = np.max(future_qs_list[index])
        new_q = reward + DISCOUNT * max_future_q
      
      current_qs = current_qs_list[index]
      current_qs[action] = new_q

      x.append(current_state)
      y.append(current_qs)
    
    self.model.fit(np.array(x)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

    if terminal_state:
      self.target_update_counter += 1

    if self.target_update_counter > UPDATE_TARGET_EVERY:
      self.target_update_counter = 0
      self.target_model.set_weights(self.model.get_weights())



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
