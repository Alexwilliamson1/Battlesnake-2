import numpy as np

#A class for storing and sampling training experiences:
class PrioritizedReplayBuffer:

    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
        self.pos = 0

    #To add an experience to the buffer:
    def push(self, state, action, reward, next_state, mask, mask_next, done):
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, mask, mask_next, done))
        else:
            #If the buffer is full, replace the oldest experience:
            self.buffer[self.pos] = (state, action, reward, next_state, mask, mask_next, done)

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity     

    #To sample experiences from the buffer according to their priority:
    def sample(self, batch_size, beta=0.4):
           
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        eps = 1e-6
        #To create sampling probabilities from the priorities:
        probs = (priorities + eps) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)
                                   
        return samples, indices, weights

    #To update the priorities according to the TD errors calculated in Training.train_step():
    def update_priorities(self, indices, priorities):
            self.priorities[indices] = np.maximum(priorities, 1e-6)

    #To return the length of the buffer:
    def __len__(self):
        return len(self.buffer)
