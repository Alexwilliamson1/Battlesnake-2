from Network import ConvolutionalNeuralNetwork
from ReplayBuffer import PrioritizedReplayBuffer
from Environment import VectorizedEnv
import Policy
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import multiprocessing as mp
import queue
import time
import random
import copy

#Hyperparameters:
gamma = 0.99
batch_size = 64
num_steps = 500_000_000
target_update = 1000
opponent_update = 5000
snapshot_interval = 250_000
beta_start = 0.4
beta_anneal_steps = 50_000_000
max_opponents = 20

#Use a CUDA device, if available, otherwise use the CPU:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

def main():
    optimizer_steps = 0
    training_steps = 0
    beta = 0.4
    #To create the policy and target networks:
    policy_net = ConvolutionalNeuralNetwork().to(device)
    target_net = ConvolutionalNeuralNetwork().to(device)

    #To create the optimizer and replay buffer:
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    buffer = PrioritizedReplayBuffer()
    
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
        training_steps = checkpoint.get('training_steps', 0)
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
        training_steps = 0
        beta = 0.4

    next_checkpoint_step = (
        (training_steps // 10_000) + 1
    ) * 10_000

    #To set the policy network to training mode and the target network to evaluation mode:
    policy_net.train()
    target_net.eval()

    # Create the main-process opponent network registry.
    opponent_networks = {}

    opponent_counter = 0

    latest_opponent_id = f"opponent_{opponent_counter}"

    opponent_networks[latest_opponent_id] = latest_opponent

    opponent_pool_ids = []

    for network in opponent_pool:

        opponent_counter += 1

        opponent_id = f"opponent_{opponent_counter}"

        opponent_networks[opponent_id] = network
        opponent_pool_ids.append(opponent_id)

    # Create a shared catalogue that all workers can access.
    manager = mp.Manager()

    opponent_catalogue = manager.dict({
        "latest_opponent_id": latest_opponent_id,
        "opponent_pool_ids": opponent_pool_ids
    })

    # Create the communication queues.
    request_queue = mp.Queue()
    transition_queue = mp.Queue()

    response_queues = [
        mp.Queue() for _ in range(10)
    ]

    # Start four worker processes.
    workers = []

    for worker_id in range(10):

        process = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                request_queue,
                response_queues[worker_id],
                transition_queue,
                opponent_catalogue
            )
        )

        process.start()
        workers.append(process)

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

    # Main-process training loop.
    pending_transitions = 0

    try:

        while training_steps < num_steps:


            policy_net.eval()

            # Service neural-network inference requests from workers.
            process_inference_requests(
                request_queue,
                response_queues,
                opponent_networks,
                policy_net,
                device
            )

            policy_net.train()

            # Receive transitions from the workers.
            try:

                transitions = transition_queue.get(timeout=0.001)

                for transition in transitions:
                    buffer.push(*transition)

                training_steps += len(transitions)
                pending_transitions += len(transitions)

            except queue.Empty:
                pass

            # Train once for approximately every 32 new transitions.
            if pending_transitions >= 32:

                trained = train_step()
                pending_transitions -= 32

                if trained:

                    optimizer_steps += 1

                    beta = min(
                        1.0,
                        beta_start
                        + (1.0 - beta_start)
                        * optimizer_steps / beta_anneal_steps
                    )

                    # Update target network.
                    if (optimizer_steps * 4) % target_update == 0:

                        target_net.load_state_dict(
                            policy_net.state_dict()
                        )

                        target_net.eval()

                    # Capture the previous latest opponent before replacing it.
                    previous_latest_id = (
                        opponent_catalogue["latest_opponent_id"]
                    )

                    # Update latest opponent.
                    if (optimizer_steps * 4) % opponent_update == 0:
                        opponent_counter += 1
                        new_opponent_id = f"opponent_{opponent_counter}"

                        opponent_networks[new_opponent_id] = (
                            make_opponent_snapshot(policy_net)
                        )

                        opponent_catalogue["latest_opponent_id"] = (
                            new_opponent_id
                        )

                    # Add the previous latest opponent to the historical pool.
                    if (optimizer_steps * 4) % snapshot_interval == 0:
                        pool_ids = list(
                            opponent_catalogue["opponent_pool_ids"]
                        )

                        if previous_latest_id not in pool_ids:
                            pool_ids.append(previous_latest_id)

                        # Keep only the most recent max_opponents snapshots.
                        pool_ids = pool_ids[-max_opponents:]

                        opponent_catalogue["opponent_pool_ids"] = pool_ids
                    if optimizer_steps % 10_000 == 0:

                        print(
                            f"Optimizer steps completed: {optimizer_steps}"
                        )

                        print(f"Beta = {beta}")

            if training_steps >= next_checkpoint_step:
                torch.save({
                    "policy_state_dict": policy_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "target_state_dict": target_net.state_dict(),

                    "latest_opponent_state_dict":
                        opponent_networks[
                            opponent_catalogue["latest_opponent_id"]
                        ].state_dict(),

                    "opponent_pool": [
                        opponent_networks[opponent_id].state_dict()
                        for opponent_id in opponent_catalogue["opponent_pool_ids"]
                    ],

                    "optimizer_steps": optimizer_steps,
                    "training_steps": training_steps,
                    "beta": beta,

                }, "checkpoint.pth")

                next_checkpoint_step += 10_000

            if training_steps and training_steps % 10_000 == 0:
                print(f"Training steps completed: {training_steps}")

    finally:

        # Stop the workers.
        for process in workers:
            process.terminate()

        for process in workers:
            process.join()

        # Save a final checkpoint.
        torch.save({
            "policy_state_dict": policy_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "latest_opponent_state_dict":
                opponent_networks[
                    opponent_catalogue["latest_opponent_id"]
                ].state_dict(),
            "opponent_pool": [
                opponent_networks[opponent_id].state_dict()
                for opponent_id in opponent_catalogue["opponent_pool_ids"]
            ],
            "optimizer_steps": optimizer_steps,
            "training_steps": training_steps,
            "beta": beta,
        }, "checkpoint.pth")

class InferenceClient:

    def __init__(self, worker_id, request_queue, response_queue):

        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.request_id = 0

    def _request(self, request_type, **kwargs):

        request_id = self.request_id
        self.request_id += 1

        self.request_queue.put({
            "worker_id": self.worker_id,
            "request_id": request_id,
            "type": request_type,
            **kwargs
        })

        while True:

            response = self.response_queue.get()

            if response["request_id"] == request_id:

                if "error" in response:
                    raise RuntimeError(response["error"])

                return response["result"]

    def select_actions(self, opponent_id, game_states):

        return self._request(
            "actions",
            opponent_id=opponent_id,
            game_states=game_states
        )

    def evaluate_values(self, observations, masks):

        return self._request(
            "values",
            observations=observations,
            masks=masks
        )

#A class for storing and choosing opponent policies:
class OpponentManager:

    def __init__(self, opponent_catalogue, historical_probability=0.20):

        self.opponent_catalogue = opponent_catalogue
        self.historical_probability = historical_probability

    def choose_opponent_id(self, rng):

        latest_id = self.opponent_catalogue["latest_opponent_id"]
        pool_ids = list(self.opponent_catalogue["opponent_pool_ids"])

        if pool_ids and rng.random() < self.historical_probability:
            return rng.choice(pool_ids)

        return latest_id

#To return a copy of the given network:
def make_opponent_snapshot(network):
    snapshot = copy.deepcopy(network)
    snapshot.eval()

    for param in snapshot.parameters():
        param.requires_grad_(False)

    return snapshot

#To return a network with the given state dictionary:
def network_from_state_dict(state_dict):

    network = ConvolutionalNeuralNetwork().to(device)

    network.load_state_dict(state_dict)
    network.eval()

    for param in network.parameters():
        param.requires_grad_(False)

    return network

def worker_process(
    worker_id,
    request_queue,
    response_queue,
    transition_queue,
    opponent_info
):

    torch.set_num_threads(1)

    inference_client = InferenceClient(
        worker_id,
        request_queue,
        response_queue
    )

    opponent_manager = OpponentManager(opponent_info)

    env = VectorizedEnv(
        num_envs=8,
        opponent_manager=opponent_manager,
        inference_client=inference_client,
        worker_id=worker_id
    )

    states, masks = env.reset()

    while True:

        # Compute MCTS actions for the eight environments.
        actions = env.mcts_actions(num_simulations=5)

        # Simulate the eight games.
        next_states, masks_next, rewards, dones, _ = env.step(actions)

        transitions = []

        for i in range(len(states)):

            transitions.append((
                states[i],
                actions[i],
                rewards[i],
                next_states[i],
                masks[i],
                masks_next[i],
                dones[i]
            ))

        # Send the transitions to the main process.
        transition_queue.put(transitions)

        states = next_states
        masks = masks_next

def process_inference_requests(
    request_queue,
    response_queues,
    opponent_networks,
    policy_net,
    device
):

    requests = []

    # Collect all currently available requests.
    while True:

        try:
            requests.append(request_queue.get_nowait())

        except queue.Empty:
            break

    if not requests:
        return

    # Group opponent-action requests by opponent ID.
    action_groups = {}

    # Collect value requests.
    value_requests = []

    for request in requests:

        if request["type"] == "actions":

            opponent_id = request["opponent_id"]

            action_groups.setdefault(
                opponent_id, []
            ).append(request)

        elif request["type"] == "values":

            value_requests.append(request)

    # Process opponent-action requests.
    for opponent_id, group in action_groups.items():

        try:
            network = opponent_networks[opponent_id]

            game_states = []

            for request in group:
                game_states.extend(request["game_states"])

            moves = Policy.select_actions_batch(
                game_states,
                network,
                device
            )
        except Exception as e:
            for request in group:
                response_queues[request["worker_id"]].put({
                    "request_id": request["request_id"],
                    "error": str(e)
                })

        offset = 0

        for request in group:

            count = len(request["game_states"])

            result = moves[offset:offset + count]

            response_queues[request["worker_id"]].put({
                "request_id": request["request_id"],
                "result": result
            })

            offset += count

    # Process policy-network value requests.
    if value_requests:

        observations = []
        masks = []

        for request in value_requests:

            observations.extend(request["observations"])
            masks.extend(request["masks"])

        states_tensor = torch.as_tensor(
            np.asarray(observations, dtype=np.float32),
            device=device
        )

        masks_tensor = torch.as_tensor(
            np.asarray(masks, dtype=np.bool_),
            device=device
        )

        with torch.inference_mode():

            q_values, _ = policy_net(states_tensor)

            q_values = q_values.masked_fill(
                ~masks_tensor,
                -torch.inf
            )

            values = q_values.max(dim=1).values.tolist()

        offset = 0

        for request in value_requests:

            count = len(request["observations"])

            result = values[offset:offset + count]

            response_queues[request["worker_id"]].put({
                "request_id": request["request_id"],
                "result": result
            })

            offset += count

if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    main()



    
