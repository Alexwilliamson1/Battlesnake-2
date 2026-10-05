# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#

import random
import typing
import torch
from Policy import policy_net, get_current_direction, mcts_action_for_games

#Information for creating a Battlesnake, including one's username and customizable snake features:
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "Alex",
        "color": "#450C0C", 
        "head": "submarine",
        "tail": "rocket",  
    }

#To start a game: 
def start(game_state: typing.Dict):
    print("A game has started.")

#To end a game:
def end(game_state: typing.Dict):
    print("A game has ended.\n")

#To compute and return a move each turn:
def move(game_state: typing.Dict) -> typing.Dict:
    with torch.inference_mode():
        action = mcts_action_for_games(
            game_state,
            policy_net,
            num_simulations=5
        )

    direction = get_current_direction(game_state)

    mapping = {
        "up":    ["left", "up", "right"],
        "down":  ["right", "down", "left"],
        "left":  ["down", "left", "up"],
        "right": ["up", "right", "down"]
    }

    move = mapping[direction][action]

    return {"move": move}

#Calling "run_server":
if __name__ == "__main__":
    from Server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})

