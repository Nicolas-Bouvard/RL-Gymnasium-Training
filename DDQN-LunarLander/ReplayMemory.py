import numpy as np

class ReplayMemory():
    def __init__(self, mem_size, state_shape, action_shape) -> None:
        self.mem_size = mem_size
        self.mem_counter = 0
        self.state_shape = state_shape
        self.state_memory = np.zeros((mem_size, state_shape), dtype=np.float32)
        self.next_state_memory = np.zeros((mem_size, state_shape), dtype=np.float32)
        self.action_shape = action_shape
        self.action_memory = np.zeros((mem_size), dtype=np.int32)
        self.reward_memory = np.zeros((mem_size), dtype=np.float32)
        self.done_memory = np.zeros((mem_size), dtype=np.bool)
    
    def store_step(self, state, action, reward, next_state, done):
        index = self.mem_counter % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index] = 1 - int(done)

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.done_memory[batch]

        return states, actions, rewards, next_states, dones