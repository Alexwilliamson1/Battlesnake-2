import numpy as np
import random
import GameSimulator
import Policy
from copy import deepcopy

#A class for saving the data related to each node in the tree:
class Node:
    def __init__(self, state, mask, observation, parent=None, action=None, rng=None, done=False, flip=False):
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
        self.flip = flip

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
def prepare_expansion(node):

    expansions = []

    valid_actions = np.where(node.mask == 1)[0]
    controlled_id = node.state["you"]["id"]

    for action in valid_actions:

        if node.flip:
            action = int(np.array([2, 1, 0])[action])
        next_state = deepcopy(node.state)
        moves = {}
        opponent_states = []

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

                move_state = GameSimulator.make_move_state(
                    next_state, i
                )

                opponent_states.append((i, move_state))

        expansions.append({
            "node": node,
            "action": action,
            "next_state": next_state,
            "moves": moves,
            "opponent_states": opponent_states
        })

    return expansions

def finish_expansion(record):

    node = record["node"]
    action = record["action"]
    next_state = record["next_state"]
    moves = record["moves"]

    GameSimulator.resolve_turn(
        next_state,
        moves,
        node.rng
    )

    controlled_id = node.state["you"]["id"]

    controlled_snake = next(
        (
            snake for snake in next_state["board"]["snakes"]
            if snake["id"] == controlled_id
        ),
        None
    )

    if controlled_snake is None:
        raise RuntimeError(
            f"There is no snake with id {controlled_id!r}."
        )

    next_state["you"] = controlled_snake

    next_state["turn"] += 1

    agent_dead = controlled_snake["health"] <= 0
    game_done = GameSimulator.game_over(next_state)
    done = agent_dead or game_done

    next_obs = Policy._get_observation(next_state)

    if done:
        next_mask = np.zeros(3, dtype=np.float32)
    else:
        next_mask = get_action_mask(next_state)

    if node.flip:
        next_obs = np.flip(next_obs, axis=2).copy()
        next_mask = next_mask[np.array([2, 1, 0])]

    node.children[action] = Node(
        state=next_state,
        mask=next_mask,
        observation=next_obs,
        parent=node,
        action=action,
        rng=node.rng,
        done=done,
        flip=node.flip
    )

#To simulate each of the agent's valid actions in the given node's state and create child nodes from the resulting data:
def expand(node, opponent_id, inference_client):

    expansions = prepare_expansion(node)

    all_opponent_states = []

    for record in expansions:
        for snake_index, move_state in record["opponent_states"]:
            all_opponent_states.append(
                (record, snake_index, move_state)
            )

    if all_opponent_states:

        game_states = [
            item[2] for item in all_opponent_states
        ]

        opponent_moves = inference_client.select_actions(
            opponent_id,
            game_states
        )

        if len(opponent_moves) != len(all_opponent_states):
            raise RuntimeError(
                "The inference process returned an incorrect "
                "number of opponent actions."
            )

        for (record, snake_index, _), move in zip(
            all_opponent_states,
            opponent_moves
        ):

            if move is None:
                record["next_state"]["board"]["snakes"][snake_index]["health"] = 0
            else:
                record["moves"][snake_index] = move

    for record in expansions:
        finish_expansion(record)

#To evaluate nodes by returning either the highest Q-value for the agent, 1 if the agent is the last surviving snake, or -1 if the agent's health is zero or will be zero after the next turn:
def evaluate_batch(nodes, inference_client):

    values = [None] * len(nodes)
    pending_indices = []

    for i, node in enumerate(nodes):

        if node.done:
            if agent_is_dead(node.state):
                values[i] = -1.0
            elif agent_has_won(node.state):
                values[i] = 1.0
            else:
                values[i] = -1.0

        elif not np.any(node.mask):
            values[i] = -1.0

        else:
            pending_indices.append(i)

    # Only send non-terminal nodes to the central inference process.
    if pending_indices:

        observations = [
            nodes[i].observation
            for i in pending_indices
        ]

        masks = [
            nodes[i].mask
            for i in pending_indices
        ]

        batch_values = inference_client.evaluate_values(
            observations,
            masks
        )

        if len(batch_values) != len(pending_indices):
            raise RuntimeError(
                "The inference process returned an incorrect "
                "number of node values."
            )

        for index, value in zip(pending_indices, batch_values):
            values[index] = float(value)

    return values


# To evaluate a single node:
def evaluate(node, inference_client):

    return evaluate_batch(
        [node],
        inference_client
    )[0]

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


