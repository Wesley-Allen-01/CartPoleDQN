from collections import deque, namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))



""" 
Instead of sampling from recent experiences, we will store each experience
in the ReplayMemory object and sample from it when training the model. This 
stabilizes the training process and lessens the negative effect of 
temporal correlations between consecutive experiences.
"""
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)