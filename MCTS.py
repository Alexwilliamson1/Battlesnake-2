import torch
import numpy as np
import random
import GameSimulator
import Policy
from copy import deepcopy

#A class for saving the data related to each node in the tree:
class Node:
    def __init__(self, state, mask, observation, parent=None, action=None, rng=None, done=False):
        self.state = deepcopy(state)
        self.observation = observation
        self.mask = mask

        self.parent = parent
        self.action = action

        self.children = {}

        self.visit_count = 0
        self.total_value = 0.0
        self.rng = rng
        self.done = done

    @property
    #To calculate the value of a node:
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count

#To select a child of the given node:
def select(node, c=1.4):
    while len(node.children) > 0:
        unvisited = [child for child in node.children.values() if child.visit_count == 0]
        #If any child nodes have a visit count of zero, randomly select one of them:
        if unvisited:
            return node.rng.choice(unvisited)

        best_score = -float("inf")
        best_child = None

        for child in node.children.values():
            #Using the Upper Confidence Bound for Trees (UCT) formula to calculate a score for each child node:
            score = (
                child.value
                + c * np.sqrt(
                    np.log(node.visit_count) /
                    child.visit_count
                )
            )

            if score > best_score:
                best_score = score
                best_child = child
        
        if best_child is None:
            raise RuntimeError("There is no \"best child\" in select().")

        node = best_child
    #Returning the child node with the highest score:
    return node

#To simulate each of the agent's valid actions in the given node's state and create child nodes from the resulting data:
def expand(node, opponent_net, device):
    valid_actions = np.where(node.mask == 1)[0]

    controlled_id = node.state["you"]["id"]

    #To loop through all safe moves in the given node's state:
    for action in valid_actions:
        action = int(action)
        next_state = deepcopy(node.state)
        moves = {}
        for i, snake in enumerate(next_state["board"]["snakes"]):
            if snake["health"] <= 0:
                continue
            if snake["id"] == controlled_id:
                current_dir = get_current_direction(next_state)

                mapping = {
                    "up": ["left", "up", "right"],
                    "down": ["right", "down", "left"],
                    "left": ["down", "left", "up"],
                    "right": ["up", "right", "down"]
                }

                moves[i] = mapping[current_dir][action]

            else:
                #To create the game state for the player at index i:
                move_state = GameSimulator.make_move_state(next_state, i)
                #To use the given opponent network to choose a move:
                move = Policy.select_action(move_state, opponent_net, device)

                if move is None:
                    snake["health"] = 0
                    continue

                moves[i] = move

        #To compute the next state, given each player's move:
        GameSimulator.resolve_turn(next_state, moves, node.rng)
        controlled_snake = next((snake for snake in next_state["board"]["snakes"] if snake["id"] == controlled_id), None)
        if controlled_snake is None:
            raise RuntimeError(f"There is no snake with id {controlled_id!r} in the simulated state.")
        next_state["you"] = controlled_snake
        agent_dead = controlled_snake["health"] <= 0
        game_done = GameSimulator.game_over(next_state)
        done = agent_dead or game_done
        next_state["turn"] += 1
        #To create the observation and action mask for the new state:
        next_obs = Policy._get_observation(next_state)
        if done:
            next_mask = np.zeros(3, dtype=np.float32)
        else:
            next_mask = get_action_mask(next_state)
        #To create a child node from the new game state:
        node.children[action] = Node(
                state = next_state,
                mask = next_mask,
                observation = next_obs,
                parent = node,
                action = action,
                rng = node.rng,
                done = done
        )

#To evaluate a node by returning either the highest Q-value for the agent, 1 if the agent is the last surviving snake, or -1 if the agent's health is zero or will be zero after the next turn:
def evaluate(node, policy_net, device):
    if agent_is_dead(node.state):
        return -1.0

    if agent_has_won(node.state):
        return 1.0

    if not np.any(node.mask):
        return -1.0

    state_tensor = (
        torch.from_numpy(node.observation)
        .float()
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():
        #Using the policy network to calculate Q-values:
        q_values, _ = policy_net(state_tensor)

    q_values = q_values.squeeze(0)

    mask = torch.as_tensor(node.mask, dtype=torch.bool, device=device)

    q_values = q_values.masked_fill(~mask, -torch.inf)
    
    #Returning the highest Q-value:
    return float(q_values.max().item())

#To traverse up the tree from the given node, incrementing each node's visit count and adding the given value to each node's total value:
def backpropagate(node, value):
    while node is not None:
        node.visit_count += 1
        node.total_value += value
        node = node.parent

#To find the agent's direction, given the game state:
def get_current_direction(state):
    snake = get_agent(state) 

    head = snake["head"]
    body = snake["body"]

    if len(body) <= 1:
        return "up"

    neck = body[1]
    #Using the head and neck coordinates to find the direction, and otherwise returning up:
    if head["x"] > neck["x"]: return "right"
    elif head["x"] < neck["x"]: return "left"
    elif head["y"] > neck["y"]: return "up"
    elif head["y"] < neck["y"]: return "down"
    return "up"

#The following function creates a list of numbers, or action mask, indicating which of the agent's moves are safe in the given state.  The number at the index of a safe move is one, and zero otherwise.
def get_action_mask(state):
    mask = np.ones(3, dtype=np.float32)

    snake = get_agent(state)   
    head = snake["head"]
    height = state["board"]["height"]
    width = state["board"]["width"]

    current_dir = get_current_direction(state)

    if current_dir == "up":
        checks = [{"x": head["x"]-1, "y": head["y"]}, {"x": head["x"], "y": head["y"]+1}, {"x": head["x"]+1, "y": head["y"]}]
    elif current_dir == "down":
        checks = [{"x": head["x"]+1, "y": head["y"]}, {"x": head["x"], "y": head["y"]-1}, {"x": head["x"]-1, "y": head["y"]}]
    elif current_dir == "left":
        checks = [{"x": head["x"], "y": head["y"]-1}, {"x": head["x"]-1, "y": head["y"]}, {"x": head["x"], "y": head["y"]+1}]
    elif current_dir == "right":
        checks = [{"x": head["x"], "y": head["y"]+1}, {"x": head["x"]+1, "y": head["y"]}, {"x": head["x"], "y": head["y"]-1}]
    
    occupied = {
        (segment["x"], segment["y"])
        for snake in state["board"]["snakes"]
        if snake["health"] > 0
        for segment in snake["body"][:-1]
    }

    for i, move in enumerate(checks):
        if move["x"] < 0 or move["x"] >= width or move["y"] < 0 or move["y"] >= height:
            mask[i] = 0.0
            continue 

        if (move["x"], move["y"]) in occupied:
            mask[i] = 0.0

    return mask

#A helper function to return the snake agent based on its id:
def get_agent(state):
    controlled_id = state["you"]["id"]
    for snake in state["board"]["snakes"]:
        if snake["id"] == controlled_id:
            return snake

    raise RuntimeError(f"There is no snake with id {controlled_id!r}.")

#To determine if the agent's health is zero:
def agent_is_dead(state):
    return get_agent(state)["health"] <= 0 

#To determine if the agent is the last surviving snake in a game:
def agent_has_won(state):
    agent = get_agent(state)
    alive = [s for s in state["board"]["snakes"] if s["health"] > 0]
    return (len(alive) == 1 and agent["health"] > 0)


