# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#

import random
import typing
import torch
from Policy import select_action, policy_net, device

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
        move = select_action(game_state, policy_net, device)
    return {"move": move}

#Calling "run_server":
if __name__ == "__main__":
    from Server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})

