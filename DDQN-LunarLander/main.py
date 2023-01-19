import gym
import time
from Agent import Agent
import gc
import tensorflow as tf
# from memory_profiler import profile

env_name = 'LunarLander-v2'
human_env = gym.make(env_name, render_mode='human')
env = gym.make(env_name)

episodes = 500
max_steps = 500

agent = Agent(8, 4, 64, 0.001, 0.99, 1, 0.01, 0.9995, 750)

step = 0
history = None

# @profile
def main():
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_step = 0
        done = False
        start_time = time.time()

        # while not done:
        for step in range(max_steps):
            action = agent.predict(state)
            next_state, reward, done, _, _ = env.step(action)

            # if step == max_steps - 1:
            #     done = True
            agent.remember(state, action, reward, next_state, done)

            history = agent.train()

            episode_reward += reward

            step += 1
            episode_step += 1
            state = next_state
            
            if done:
                break

            print('Episode %i: step %i, duration %.1fs, episode reward %.1f, loss %.3f...' % (episode, step, time.time() - start_time, \
                episode_reward, history.history['loss'][0] if history is not None else 0), end='\r')
        
        print('Episode %i finished with %i steps in %.2fs: score %i, exploration rate %.2f, loss %.3f, samples %i' % \
            (episode, episode_step, time.time() - start_time, episode_reward, \
                agent.exploration_rate, history.history['loss'][0] if history is not None else 0, \
                min(agent.memory.mem_counter, agent.memory.mem_size)))
                
        # if agent.counter > 0:
        #     agent.update_exploration_rate()
        
        # Running visual example
        if episode % 5 == 0:
            done = False
            state, _ = human_env.reset()
            episode_reward = 0
            for step in range(max_steps):
                action = agent.predict(state, training=False)
                state, reward, done, _, _ = human_env.step(action)
                episode_reward += reward
                time.sleep(0.02)

                if done:
                    break
            print('/!\ Test episode /!\ Episode %i: score %i' % \
            (episode, episode_reward))
        
        # Clean garbage collector to fix memory leaks
        _ = gc.collect()
        tf.keras.backend.clear_session()

if __name__ == '__main__':
    main()
    env.close()
    human_env.close()