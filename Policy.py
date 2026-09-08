import torch   
import numpy as np
import random
from ResidualNetwork import ResidualNetwork

device = torch.device("cpu")
#Creating the policy network and using the saved state dictionary in checkpoint.pth:
policy_net = ResidualNetwork().to(device)

try:
    checkpoint = torch.load(
        "checkpoint.pth",
        map_location=device,
        weights_only=False
    )

    policy_net.load_state_dict(checkpoint["policy_state_dict"])
    print("The live policy network loaded successfully.")

except (FileNotFoundError, KeyError, RuntimeError) as error:
    print("The live policy network failed to load.")
    print(error)

policy_net.eval()

#To create an observation (2-D array), given the game state, with a channel (array) for each of the following: the coordinates of the agent's head, the coordinates of the agent's body segments, the agent's health / 100, the agent's length / the board area, the coordinates of the heads of opponent snakes, the coordinates of the body segments of opponent snakes, the coordinates of food objects, the length of the longest snake / the board area, the game turn / 100, and the coordinates of hazards:
def _get_observation(state):
    
    height = state["board"]["height"]
    width = state["board"]["width"]
    snakes = state["board"]["snakes"]
   
    controlled_id = state["you"]["id"]
    agent_index = None
    for i, snake in enumerate(snakes):
        if snake["id"] == controlled_id:
            agent_index = i
            break

    if agent_index is None: 
        raise ValueError(f'There is no snake with id {controlled_id!r}.')

    #To create the 2-D array with initial values of zero:
    grid = np.zeros((10, height, width), dtype=np.float32)

    max_length = 1

    for i, snake in enumerate(snakes):
        if snake["health"] <= 0:
            continue

        length = snake["length"]
        max_length = max(max_length, length)

        if i == agent_index:
            x, y = snake["head"]["x"], snake["head"]["y"]
            y_flipped = height - 1 - y
            grid[0, y_flipped, x] = 1.0
            grid[2, :, :] = snake["health"] / 100.0
            grid[3, :, :] = snake["length"] / (width * height)
            for j, segment in enumerate(snake["body"]):
                x, y = segment["x"], segment["y"]
                y_flipped = height - 1 - y
                grid[1, y_flipped, x] = (length - j) / float(length)
        else:
            x, y = snake["head"]["x"], snake["head"]["y"]
            y_flipped = height - 1 - y
            grid[4, y_flipped, x] = 1.0
            for k, segment in enumerate(snake["body"]):
                x, y = segment["x"], segment["y"]
                y_flipped = height - 1 - y
                grid[5, y_flipped, x] = (length - k) / float(length)

    for food in state["board"]["food"]:
        y_flipped = height - 1 - food["y"]
        grid[6, y_flipped, food["x"]] = 1.0

    grid[7, :, :] = max_length / (width * height)
    grid[8, :, :] = state["turn"] / 100.0

    for hazard in state["board"]["hazards"]:
        y_flipped = height - 1 - hazard["y"]
        grid[9, y_flipped, hazard["x"]] = 1.0

    direction = get_current_direction(state)

    #Rotating the observation so that the agent is moving up:
    if direction == "left":
        grid = np.rot90(grid, k=-1, axes=(1, 2))
    elif direction == "right":
        grid = np.rot90(grid, k=1, axes=(1, 2))
    elif direction == "down":
        grid = np.rot90(grid, k=2, axes=(1, 2))

    return np.ascontiguousarray(grid, dtype=np.float32)

#The following function uses the policy network to calculate three Q-values, then returns the highest Q-value that corresponds to a safe move.  If all moves result in the agent's health being zero, the function returns the move corresponding to the highest Q-value:
def select_action(game_state, network, device):
    current_dir = get_current_direction(game_state)

    mapping = {
        "up": ["left", "up", "right"],
        "down": ["right", "down", "left"], 
        "left": ["down", "left", "up"],
        "right": ["up", "right", "down"]
    }
    state = _get_observation(game_state)
    state_t = torch.from_numpy(state).unsqueeze(0).float().to(device)

    with torch.no_grad():
        q_values, _ = network(state_t)

    action_order = torch.argsort(q_values[0], descending=True)
    candidate_moves = mapping[current_dir]

    for action_index in action_order:
        move = candidate_moves[action_index.item()]

        if is_safe_move(game_state, move):
            return move

    return candidate_moves[action_order[0].item()]

#To return the direction of the agent, given the game state:
def get_current_direction(state):
    snakes = state["board"]["snakes"]
    controlled_id = state["you"]["id"]

    this_snake = next((snake for snake in snakes if snake["id"] == controlled_id), None)
    if this_snake is None:
        raise ValueError(f'There is no snake with id {controlled_id!r}.')

    body = this_snake["body"]
    head = this_snake["head"]

    if len(body) <= 1:
        return "up"

    neck = body[1]
    #Using the coordinates of the agent's head and neck to calculate it's direction:
    if head["x"] > neck["x"]: return "right"
    elif head["x"] < neck["x"]: return "left"
    elif head["y"] > neck["y"]: return "up"
    elif head["y"] < neck["y"]: return "down"
    
    return "up"

#To determine if a move is within the boundaries of the game board and is onto a free board square:
def is_safe_move(game_state, move):
    head = game_state["you"]["head"]
    width = game_state["board"]["width"]
    height = game_state["board"]["height"]

    x, y = head["x"], head["y"]

    if move == "up":
        y += 1
    elif move == "down":
        y -= 1
    elif move == "left":
        x -= 1
    elif move == "right":
        x += 1

    if x < 0 or x >= width or y < 0 or y >= height:
        return False
    
    #Creating a list of game squares occupied by snake bodies, excluding their tails:
    occupied = {
        (segment["x"], segment["y"])
        for snake in game_state["board"]["snakes"]
        if snake["health"] > 0
        for segment in snake["body"][:-1]
    }

    return (x, y) not in occupied
