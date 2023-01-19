import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from ReplayMemory import ReplayMemory
import random
import numpy as np

env_name = 'CartPole-v1'
# env = gym.make(env_name, render_mode='human')
env = gym.make(env_name)

num_epochs = 100
max_steps = 500

learning_rate = 0.005
discount_rate = 0.99

batch_size = 32

min_exploration_rate = 0.01
max_exploration_rate = 1
exploration_rate = max_exploration_rate
exploration_rate_decay = 0.995

agent = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    # tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear')
])

next_agent = tf.keras.Sequential([agent])

agent.compile(loss=tf.losses.mean_squared_error,optimizer=tf.optimizers.Adam(learning_rate=learning_rate))

memory = ReplayMemory(1000000, 4, 2)

for epoch in range (num_epochs):
    state, _ = env.reset()
    epoch_reward = 0
    random_action_used = 0
    prediction_used = 0

    for _ in range(max_steps):

        if random.uniform(0, 1) > exploration_rate:
            action = np.argmax(agent.predict(state[np.newaxis,:], verbose=0))
            prediction_used += 1
        else:
            action = env.action_space.sample()
            random_action_used += 1

        new_state, reward, done, _, _ = env.step(action)

        memory.store_step(state, action, reward, new_state, int(done))

        state = new_state
        epoch_reward += reward

        # if reward < 1 or done:
        #     print('Reward is ', str(reward), 'and done is ', str(done))

        if memory.mem_counter < batch_size:
            if done:
                break
            # print('Epoch %i: not enough data to train. score %i' % (epoch, epoch_reward))
            continue

        # exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_rate_decay * epoch)
        exploration_rate = max(exploration_rate * exploration_rate_decay, min_exploration_rate)

        states, actions, rewards, next_states, dones = memory.sample_buffer(batch_size)

        action_values = np.array([0, 1], dtype=np.int8)
        action_indexes = np.dot(actions, action_values).astype(int)
        
        targets = agent.predict(states, verbose=0)
        # next_predictions = next_agent.predict(next_states, verbose=0)
        next_predictions = agent.predict(next_states, verbose=0)

        batch_indexes = np.arange(batch_size, dtype=np.int32)

        targets[batch_indexes, action_indexes] = reward + discount_rate * np.max(next_predictions, axis=1) * dones

        history = agent.fit(states, targets, epochs=1, verbose=0)

        if done:
            break
    

    # If true, it means that there is not enough samples in memory to train.
    
    if memory.mem_counter > batch_size:
        print('Epoch %i: loss %.4f, score %i, exploration rate %.2f, random action used %i over %i actions    ' % \
            (epoch, history.history['loss'][0], epoch_reward, exploration_rate, random_action_used, random_action_used + prediction_used))

print()

env.close()
