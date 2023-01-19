import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import random
from ReplayMemory import ReplayMemory
from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import load_model

class Agent():
    def __init__(self, state_shape, action_shape, batch_size, learning_rate, discount, \
        exploration_rate, min_exploration_rate, exploration_rate_decay, ddqn_copy_frequency) -> None:
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.memory = ReplayMemory(100000, state_shape, action_shape)
        self.batch_size = batch_size
        self.batch_indexes = np.arange(batch_size, dtype=np.int32)
        self.action_values = np.array([i for i in range(self.action_shape)], dtype=np.int8)
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.counter = 0
        self.ddqn_copy_frequency = ddqn_copy_frequency
        self.model = self._build_model(state_shape, action_shape)
        self.target_model = Sequential([self.model])

    def train(self):
        if self.memory.mem_counter < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)

        action_indexes = self.action_values[actions]

        predictions = self.model(tf.convert_to_tensor(states), training=False).numpy()
        # predictions = self.model.predict(states, verbose=0)
        next_predictions = self.target_model(tf.convert_to_tensor(next_states), training=False).numpy()
        # next_predictions = self.target_model.predict(next_states, verbose=0)
        
        target = predictions.copy()
        target[self.batch_indexes, action_indexes] = rewards + self.discount * np.max(next_predictions, axis=1) * dones
        
        history = self.model.fit(states, target, epochs=1, verbose=0)

        self.update_exploration_rate()

        self.counter += 1
        if self.counter % self.ddqn_copy_frequency == 0:
            self.target_model = Sequential([self.model])
        
        return history

    def predict(self, state, training = True):
        if not training or random.uniform(0, 1) > self.exploration_rate:
            # return np.argmax(self.model.predict(state[np.newaxis,:], verbose=0))
            return np.argmax(self.model(tf.convert_to_tensor(state[np.newaxis,:]), training=False).numpy())
        else:
            return np.random.choice(self.action_shape)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.store_step(state, action, reward, next_state, done)
    
    def update_exploration_rate(self):
        self.exploration_rate = max(self.exploration_rate * self.exploration_rate_decay, self.min_exploration_rate)
    
    def _build_model(self, state_shape, action_shape):
        model = Sequential()
        model.add(Dense(512, input_dim=state_shape, activation=relu))
        model.add(Dense(256, activation=relu))
        model.add(Dense(action_shape, activation=linear))
        model.compile(loss=mean_squared_error,optimizer=Adam(lr=self.learning_rate))
        print(model.summary())

        return model