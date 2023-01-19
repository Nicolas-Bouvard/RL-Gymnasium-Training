import gym
import numpy as np
import random
import time

env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=False)

num_epochs = 50000
max_steps = 20

learning_rate = 0.01
discount_rate = 0.99

max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_rate = max_exploration_rate
exploration_decay_rate = 0.001

all_epochs_rewards = []

q_table = np.zeros((env.observation_space.n, env.action_space.n))

print('Starting learning')
for epoch in range(num_epochs):
    print('Learning with an exploration rate at %.2f' % exploration_rate, 'in progress...', str(int((epoch + 1) / num_epochs * 100)) + '%', end='\r')
    state, _ = env.reset()

    epoch_reward = 0
    done = False

    for step in range(max_steps):
        if random.uniform(0, 1) > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()
        
        new_state, reward, done, _, _ = env.step(action)

        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state,:]))

        state = new_state
        epoch_reward += reward

        if done:
            break
    
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * epoch)

    all_epochs_rewards.append(epoch_reward)

print()
print('Learning session finished.')

rewards_per_thousand_epochs = np.split(np.array(all_epochs_rewards), num_epochs / 5000)
index = 0
for epoch_reward in rewards_per_thousand_epochs:
    print('Epoch', (index + 1) * 5000, 'average reward:', sum(epoch_reward / 5000))
    index += 1

input('Click to start test games...')

env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=False, render_mode='human')

for epoch in range(1):
    state, _ = env.reset()
    env.render()
    time.sleep(0.3)
    done = False

    for step in range(max_steps):
        action = np.argmax(q_table[state,:])
        state, reward, done, _, _ = env.step(action)
        env.render()
        time.sleep(0.3)
        if done:
            break

print('Finished demo session.')

env.close()

print('All done.')