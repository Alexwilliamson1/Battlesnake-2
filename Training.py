from ResidualNetwork import ResBlock, ResidualNetwork
from ReplayBuffer import PrioritizedReplayBuffer
from Environment import Environment, VectorizedEnv
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

#Hyperparameters:
gamma = 0.99
batch_size = 64
num_steps = 100000000
target_update = 1000
opponent_update = 5000
snapshot_interval = 10000
optimizer_steps = 0
beta = 0.4
beta_start = 0.4
beta_anneal_steps = 10_000_000
max_opponents = 20

#Use a CUDA device, if available, otherwise use the CPU:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

#To create the policy and target networks:
policy_net = ResidualNetwork().to(device)
target_net = ResidualNetwork().to(device)

#To create the optimizer and replay buffer:
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
buffer = PrioritizedReplayBuffer()

#A class for storing and choosing opponent policies:
class OpponentManager:
    def __init__(self, latest_opponent, opponent_pool=None, historical_probability=0.20):
        self.latest_opponent = latest_opponent
        self.opponent_pool = (opponent_pool if opponent_pool is not None else [])
        self.historical_probability = historical_probability

    #To return the current opponent policy with an 80% probability, and otherwise return a randomly chosen policy from the pool of opponent policies:
    def choose_opponent(self, rng):
        if (self.opponent_pool and rng.random() < self.historical_probability):
            return rng.choice(self.opponent_pool)
        return self.latest_opponent

#To return a copy of the given network:
def make_opponent_snapshot(network):
    snapshot = copy.deepcopy(network)
    snapshot.eval()

    for param in snapshot.parameters():
        param.requires_grad_(False)

    return snapshot

#To return a network with the given state dictionary:
def network_from_state_dict(state_dict):
    network = ResidualNetwork().to(device)
    network.load_state_dict(state_dict)
    network.eval()

    for param in network.parameters():
        param.requires_grad_(False)

    return network

#To assign the state dictionaries and other data saved in checkpoint.pth to ResidualNetwork objects and other variables:
try: 
    checkpoint = torch.load("checkpoint.pth", map_location=device)
    policy_net.load_state_dict(checkpoint['policy_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'target_state_dict' in checkpoint:
        target_net.load_state_dict(checkpoint['target_state_dict'])
    else:
        target_net.load_state_dict(policy_net.state_dict())
    if 'latest_opponent_state_dict' in checkpoint:
        latest_opponent = network_from_state_dict(checkpoint['latest_opponent_state_dict'])
    elif 'opponent_state_dict' in checkpoint:
        latest_opponent = network_from_state_dict(checkpoint['opponent_state_dict'])
    else:
        latest_opponent = make_opponent_snapshot(policy_net)
    saved_pool = checkpoint.get('opponent_pool', [])
    opponent_pool = [network_from_state_dict(state) for state in saved_pool]
    optimizer_steps = checkpoint.get('optimizer_steps', 0)
    beta = checkpoint.get('beta', 0.4)

    print("checkpoint.pth has succesfully loaded.")

#If checkpoint.pth fails to load, initialize the target and opponent networks from the policy network and reset the other saved variables:
except Exception as e:
    print("checkpoint.pth failed to load.")
    print(e)

    target_net.load_state_dict(policy_net.state_dict())
    latest_opponent = make_opponent_snapshot(policy_net) 
    opponent_pool = []
    optimizer_steps = 0
    beta = 0.4

#To set the policy network to training mode and the target network to evaluation mode:
policy_net.train()
target_net.eval()

#To create an OpponentManager object, given the current opponent policy, opponent policy pool, and the probability used for choosing an opponent policy:
opponent_manager = OpponentManager(latest_opponent=latest_opponent, opponent_pool=opponent_pool, historical_probability=0.20)

#To create a VectorizedEnvironment object, given the number of environments, an OpponentManager object, and a device:
env = VectorizedEnv(num_envs=8, opponent_manager=opponent_manager, device=device)

#To create tensors from the training data; calculate Q-values, target values, TD errors, loss, and priorities; and step the optimizer:
def train_step():
    #Accumulating 2000 experiences in the replay buffer before training the network:
    if len(buffer) < 2000:
        return False

    #Obtaining a sample of experiences from the replay buffer:
    batch, indices, weights = buffer.sample(batch_size, beta)

    states, actions, rewards, next_states, masks, masks_next, dones = zip(*batch)

    #Creating tensors from the experience data:
    states_np = np.array(states, dtype=np.float32)
    states_tensor = torch.from_numpy(states_np).float().to(device)

    actions_np = np.array(actions, dtype=np.int64)
    actions_tensor = torch.from_numpy(actions_np).long().to(device)

    rewards_np = np.array(rewards, dtype=np.float32)
    rewards_tensor = torch.from_numpy(rewards_np).float().to(device)

    next_states_np = np.array(next_states, dtype=np.float32)
    next_states_tensor = torch.from_numpy(next_states_np).float().to(device)

    masks_np = np.array(masks, dtype=np.float32)
    masks_tensor = torch.from_numpy(masks_np).float().to(device)

    masks_next_np = np.array(masks_next, dtype=np.float32)
    masks_next_tensor = torch.from_numpy(masks_next_np).float().to(device)

    dones_np = np.array(dones, dtype=np.float32)
    dones_tensor = torch.from_numpy(dones_np).float().to(device)

    weights_tensor = torch.from_numpy(weights).float().to(device)

    #Calculating Q-values for each state
    q_values_all, _ = policy_net(states_tensor)
    q_values = q_values_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        #Using the policy network to calculate Q-values for the next states:
        next_q_policy, _ = policy_net(next_states_tensor)
        #Applying the masks for the next states:
        next_q_policy = next_q_policy.masked_fill(masks_next_tensor == 0, -1e9)
        #Finding the highest Q-value for each next state and returning its index:
        next_actions = next_q_policy.argmax(dim=1, keepdim=True)
        #Using the target network to calculate Q-values for the next states:
        next_q_target, next_v_target = target_net(next_states_tensor)
        #Applying the masks for the next states:
        next_q_target = next_q_target.masked_fill(masks_next_tensor == 0, -1e9)
        #Using the target network Q-values corresponding to the "next_actions" indices as the Q-values for the next states:
        next_q = next_q_target.gather(1, next_actions).squeeze(1)
        #Setting all Q-values for terminal states to zero:
        next_q = next_q * (1 - dones_tensor)
        #Calculating the target values by adding the rewards to the product of the Q-values for the next states and the discount factor:
        target = rewards_tensor + gamma * next_q
        
    #Calculating the temporal difference (TD) errors:
    td_errors = q_values - target
    #Calculating the average loss from the loss values for each sampled experience:
    q_loss_per_sample = F.smooth_l1_loss(q_values, target, reduction='none')
    q_loss = (weights_tensor * q_loss_per_sample).mean()
    loss = q_loss 
    #Ensuring all loss values are finite:
    if not torch.isfinite(loss).item():
        print("Skipping a nonfinite loss:", loss.item())
        optimizer.zero_grad(set_to_none=True)
        return False
    optimizer.zero_grad(set_to_none=True)
    #Determining which network weights contribute to the loss through backpropogation:
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    #Updating the network weights:
    optimizer.step()
    #Creating new priority values for buffer sampling from the TD errors:
    new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
    #Updating the current priority values:
    buffer.update_priorities(indices, new_priorities)
    return True

#Resetting the environments (initializing games):
states, masks = env.reset()

#The training loop:
for step in range(num_steps):
    if step % 500 == 0:
        #Regularly saving training data in checkpoint.pth:
        torch.save({
            'policy_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'target_state_dict': target_net.state_dict(),
            'latest_opponent_state_dict': opponent_manager.latest_opponent.state_dict(),
            'opponent_pool':[network.state_dict() for network in opponent_manager.opponent_pool],
            'optimizer_steps': optimizer_steps,
            'beta': beta,
        }, "checkpoint.pth")

    policy_net.eval()
    #Computing an action for each environment:
    actions = [environment.mcts_action(policy_net) for environment in env.envs]
    policy_net.train()
    #Stepping each environment by simulating the actions and returning the transition data:
    next_states, masks_next, rewards, dones, _ = env.step(actions)
    for i in range(len(states)):
        #Adding new experiences to the replay buffer:
        buffer.push(states[i], actions[i], rewards[i], next_states[i], masks[i], masks_next[i], dones[i])
    states = next_states 
    masks = masks_next
    #Training the network:
    trained = train_step()
    if trained:
        #Incrementing optimizer_steps and beta for each network update:
        optimizer_steps += 1
        beta = min(1.0, beta_start + (1.0 - beta_start) * optimizer_steps / beta_anneal_steps)

        #Updating the target network every 1000 optimizer steps:
        if optimizer_steps % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()

        #Updating the opponent network every 5000 otimizer steps:
        if optimizer_steps % opponent_update == 0:
            previous_latest = opponent_manager.latest_opponent

            opponent_manager.latest_opponent = (make_opponent_snapshot(policy_net))

            #Adding the previous opponent network to the opponent pool every 10000 optimizer steps:
            if optimizer_steps % snapshot_interval == 0:
                opponent_manager.opponent_pool.append(previous_latest)

                #Removing the oldest opponent network from the pool when the pool size exceeds 20:
                if len(opponent_manager.opponent_pool) > max_opponents:
                    opponent_manager.opponent_pool.pop(0)

        #Outputting training data:
        if optimizer_steps % 10000 == 0:
            print(f"Optimizer steps completed: {optimizer_steps}")
            print(f"Beta = {beta}")

    if (step * 8) % 400 == 0:
        print(f"Training steps completed: {step * 8}")
    


    
