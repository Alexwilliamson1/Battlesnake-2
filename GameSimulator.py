import copy
import random
import uuid
#import maxn
#import safe_random_bot
#import beginner_bot

DIRECTIONS = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

#The following function creates a game state for a new game, given the number of players, board dimensions, and seed for the random number generator.  Uncomment the "royale" field to run games in Royale mode.
def init_game(num_snakes, width=11, height=11, rng=None):
    if rng is None:
        rng = random.Random()

    game_id = uuid.uuid4().hex

    snakes = []
    mid_x = width // 2
    mid_y = height // 2
    spawn_points = [
        (1, height - 2),
        (width - 2, 1),
        (width - 2, height - 2),
        (1, 1),
        (mid_x, height - 2),
        (mid_x, 1),
        (1, mid_y),
        (width - 2, mid_y),     
    ]

    for i in range(num_snakes):
        x, y = spawn_points[i % len(spawn_points)]
        snake = {
            "id": f"snake-{i}",
            "name": f"Snake {i}",
            "health": 100,
            "body": [
                {"x": x, "y": y},
                {"x": x, "y": y},
                {"x": x, "y": y}
            ],
            "head": {"x": x, "y": y},
            "length": 3,
        }
        snakes.append(snake)

    state = {
        "game": {
            "id": game_id,
            "ruleset": {
                "name": "standard",
                "settings": {
                    "foodSpawnChance": 15,
                    "minimumFood": 1,
                    "hazardDamagePerTurn": 14,
                    #"royale": {
                        #"shrinkEveryNTurns": 10,
                    #}
                },
            },
        },
        "turn": 0,
        "board": {
            "width": width,
            "height": height,
            "food": [
                {"x": 5, "y": 5},
                {"x": 0, "y": 8},
                {"x": 10, "y": 2},
                {"x": 2, "y": 0},
                {"x": 8, "y": 10},
            ],
            "hazards": [],
            "snakes": snakes,
        },
        "royale": {
            "min_x": 0,
            "max_x": width - 1, 
            "min_y": 0,
            "max_y": height - 1,
        }
    }

    spawn_food(state, rng)
    return state

#To add food objects to random free board squares up to the amount set by the "minimumFood" field in the game state:
def spawn_food(state, rng):
    board = state["board"]
    width = board["width"]
    height = board["height"]

    occupied = compute_occupied(board)

    free_cells = [
        (x, y) 
        for x in range(width) 
        for y in range(height) 
        if (x, y) not in occupied
    ]

    while len(board["food"]) < state["game"]["ruleset"]["settings"]["minimumFood"] and free_cells:
        x, y = rng.choice(free_cells)
        board["food"].append({"x": x, "y": y})
        free_cells.remove((x, y))

#To add a food object to a random free board square:
def spawn_random_food(state, rng):
    board = state["board"]
    width = board["width"]
    height = board["height"]

    occupied = compute_occupied(board)
    
    free_cells = [
        (x, y) 
        for x in range(width) 
        for y in range(height) 
        if (x, y) not in occupied
    ]

    if free_cells:
        x, y = rng.choice(free_cells)
        board["food"].append({"x": x, "y": y})
        free_cells.remove((x, y))

#The following function runs a game by generating moves and updating the game state until zero or one players remain alive.  It returns the index of the winning snake and a list of the number of turns survived by each snake player.
def run_game(genomes, max_turns=500, seed=None, logger=None):
    rng = random.Random(seed)
    num_players = 4
    state = init_game(num_players, rng=rng)
    turns = [0] * num_players
    winner_index = -1

    while not game_over(state) and state["turn"] < max_turns:
        moves = {}

        for i, snake in enumerate(state["board"]["snakes"]):
            if snake["health"] <= 0:
                continue

            move_state = make_move_state(state, i)
            if (i < 2):
                move = maxn.find_best_move(move_state, i, genomes[i])
            elif (i == 2):
                move = safe_random_bot.safe_random_bot(move_state, i)
            elif (i == 3):
                move = beginner_bot.beginner_bot(move_state, i)

            if move is None:
                snake["health"] = 0
                continue
            moves[i] = move
            turns[i] += 1

            if logger:
                logger.log_turn(move_state, i, move)

        resolve_turn(state, moves, rng)
        state["turn"] += 1

    alive = [i for i, snake in enumerate(state["board"]["snakes"]) if snake["health"] > 0]
    
    if len(alive) == 1:
        winner_index = alive[0]

    if logger:
        logger.log_result(state)

    return winner_index, turns

#To compute a new game state each turn:
def resolve_turn(state, moves, rng):
    board = state["board"]
    snakes = board["snakes"]
    width, height = board["width"], board["height"]
    hazard_damage = state["game"]["ruleset"]["settings"]["hazardDamagePerTurn"]
    hazard_set = {(h["x"], h["y"]) for h in board["hazards"]}

    #To update each snake's body coordinates and health:
    for i, snake in enumerate(snakes):
        if snake["health"] <= 0 or i not in moves: continue
        
        dx, dy = DIRECTIONS[moves[i]]
        new_head = {"x": snake["head"]["x"] + dx, "y": snake["head"]["y"] + dy}
        
        snake["body"].insert(0, new_head)
        snake["head"] = new_head
        snake["body"].pop()
        snake["health"] -= 1

    for snake in snakes:
        if snake["health"] <= 0:
            continue
        if (snake["head"]["x"], snake["head"]["y"]) in hazard_set:
            snake["health"] -= hazard_damage

    #To update a snake's length and health if it collides with a food object:
    food_to_remove = []
    for food in board["food"]:
        eating_snakes = [s for s in snakes if s["health"] > 0 and s["head"] == food]
        if eating_snakes:
            food_to_remove.append(food)
            for s in eating_snakes:
                s["health"] = 100
                s["body"].append(copy.deepcopy(s["body"][-1]))
                s["length"] += 1
    
    #To remove a food object if it has been "consumed":
    board["food"] = [f for f in board["food"] if f not in food_to_remove]

    #To add at least "minimumFood" number of food objects to the board:
    spawn_food(state, rng)
    if rng.randint(1, 100) <= state["game"]["ruleset"]["settings"]["foodSpawnChance"]:
        spawn_random_food(state, rng)

    #To add and update the hazard zone if one is running games in Royale mode:
    settings = state["game"]["ruleset"]["settings"]

    #if "royale" in settings:
        #interval = settings["royale"]["shrinkEveryNTurns"]
        #if state["turn"] % interval == 0 and state["turn"] > 0:
            #shrink_royale_zone(state)

        #update_royale_hazards(state)

    #To remove snakes from the board if they move beyond the board boundaries, their health value is zero, or they collide with their own body or an opponent of equal or greater length: 
    dead_indices = set()
    for i, s in enumerate(snakes):
        if s["health"] <= 0: continue
        
        head = s["head"]
        if head["x"] < 0 or head["x"] >= width or head["y"] < 0 or head["y"] >= height:
            dead_indices.add(i)
        
        elif s["health"] <= 0:
            dead_indices.add(i)

    for i, s1 in enumerate(snakes):
        for j, s2 in enumerate(snakes):
            if i >= j:
                continue
            if s1["health"] <= 0 or s2["health"] <= 0:
                continue

            if s1["head"] == s2["head"]:
                if s1["length"] > s2["length"]:
                    dead_indices.add(j)
                elif s2["length"] > s1["length"]:
                    dead_indices.add(i)
                else:
                    dead_indices.add(i)
                    dead_indices.add(j)

            elif s1["head"] in s2["body"][1:]:
                dead_indices.add(i)

    for i in dead_indices:
        snakes[i]["health"] = 0

#To determine if zero or one snakes are alive:
def game_over(state):
    alive = [s for s in state["board"]["snakes"] if s["health"] > 0]
    return len(alive) <= 1

#To add the "you" field to the game state:
def make_move_state(state, snake_index):
    board_copy = copy.deepcopy(state["board"])
    return {
        "game": state["game"],
        "turn": state["turn"],
        "board": board_copy,
        "you": board_copy["snakes"][snake_index],
    }

#To create a list of board coordinates where a snake, food object, or hazard are located:
def compute_occupied(board):
    occupied = set()
    for snake in board["snakes"]:
        for segment in snake["body"]:
            occupied.add((segment["x"], segment["y"])) 
    for f in board["food"]:
        occupied.add((f["x"], f["y"]))
    for h in board["hazards"]:
        occupied.add((h["x"], h["y"]))
    return occupied

#To update the hazard area in the game state:
def update_royale_hazards(state):
    royale = state.get("royale")
    if royale is None:
        return

    board = state["board"]
    width = board["width"]
    height = board["height"]

    min_x = royale["min_x"]
    max_x = royale["max_x"]
    min_y = royale["min_y"]
    max_y = royale["max_y"]

    hazards = []

    for x in range(width):
        for y in range(height):
            if x < min_x or x > max_x or y < min_y or y > max_y:
                hazards.append({"x": x, "y": y})

    board["hazards"] = hazards

#To modify the boundaries of the hazard area:
def increase_royale_area(state):
    royale = state["royale"]

    if royale["min_x"] < royale["max_x"]:
        royale["min_x"] += 1
        royale["max_x"] -= 1

    if royale["min_y"] < royale["max_y"]:
        royale["min_y"] += 1
        royale["max_y"] -= 1
