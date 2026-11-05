import torch
import random
import numpy as np
import GameSimulator
import MCTS
import Policy

#An array of the action indices in reverse order:
FLIP = np.array([2, 1, 0])

#The following is a class for creating and calling functions on a vector of environment objects.  A VectorizedEnv object is initialized with a number of environments, an OpponentManager object, and a device, such as "cpu."
class VectorizedEnv:
    def __init__(
        self,
        num_envs,
        opponent_manager,
        inference_client,
        worker_id
    ):
        self.envs = [
            Environment(
                opponent_manager=opponent_manager,
                device=torch.device("cpu")
            )
            for _ in range(num_envs)
        ]

        self.num_envs = num_envs
        self.inference_client = inference_client
        self.worker_id = worker_id

        self.games_complete = 0
        self.reward = 0

    #To reset each environment by 
    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, masks = zip(*results)
        return np.stack(obs), np.stack(masks)

    #To simulate turns, initialize new games, and return the observation, action mask, reward, and done indicator for the transition in each environment:
    def step(self, actions):
        if len(actions) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} actions, got {len(actions)}"
            )

        # Phase 1: Prepare each environment's controlled-snake move.
        all_moves = []
        # Group opponent requests by opponent ID.
        states_by_opponent = {}

        for env_index, (env, action) in enumerate(zip(self.envs, actions)):
            moves, opponent_states = env.prepare_step(action)
            all_moves.append(moves)

            for snake_index, move_state in opponent_states:

                opponent_id = env.opponent_id

                if opponent_id not in states_by_opponent:
                    states_by_opponent[opponent_id] = []

                states_by_opponent[opponent_id].append(
                    (move_state, env_index, snake_index)
                )

        # Request opponent actions from the main process.
        for opponent_id, entries in states_by_opponent.items():

            game_states = [entry[0] for entry in entries]

            opponent_actions = self.inference_client.select_actions(
                opponent_id,
                game_states
            )

            for entry, move in zip(entries, opponent_actions):
                _, env_index, snake_index = entry

                if move is None:
                    self.envs[env_index].state["board"]["snakes"][snake_index]["health"] = 0
                else:
                    all_moves[env_index][snake_index] = move

        # Phase 3: Resolve all turns.
        results = [
            env.finish_step(moves)
            for env, moves in zip(self.envs, all_moves)
        ]

        obs, masks, rewards, dones, infos = zip(*results)

        obs = list(obs)
        masks = list(masks)

        # Preserve your original completed-game tracking and reset logic.
        for i, done in enumerate(dones):
            if done:
                new_obs, new_mask = self.envs[i].reset()
                obs[i] = new_obs
                masks[i] = new_mask

                self.games_complete += 1
                self.reward += rewards[i]

                if self.games_complete % 100 == 0:
                    print(
                        "Average reward (100 games): ",
                        self.reward / 100
                    )
                    self.reward = 0

                if self.games_complete % 1000 == 0:
                    print(f"{self.games_complete} games have run.")

        return (
            np.stack(obs),
            np.stack(masks),
            np.array(rewards),
            np.array(dones),
            infos
        )

    def mcts_actions(self, num_simulations=5):

        roots = []

        for env in self.envs:

            observation = env._apply_flip(env._get_observation())

            mask = MCTS.get_action_mask(env.state)

            if env.flip:
                mask = mask[FLIP]

            root = MCTS.Node(
                env.state,
                mask,
                observation=observation,
                rng=env.mcts_rng,
                flip=env.flip
            )

            roots.append(root)

        try:

            for _ in range(num_simulations):

                selected_nodes = []
                expansion_records = []

                # 1. Select nodes and prepare expansions.
                for env, root in zip(self.envs, roots):

                    rng = env.mcts_rng
                    node = MCTS.select(root)

                    if node.done:
                        pass

                    elif (
                        node.visit_count == 0
                        and node.parent is not None
                    ):
                        pass

                    elif len(node.children) == 0:

                        records = MCTS.prepare_expansion(node)

                        for record in records:
                            record["opponent_id"] = env.opponent_id

                        expansion_records.extend(records)

                    selected_nodes.append(node)

                # 2. Collect all opponent observations.
                opponent_requests = []

                for record in expansion_records:

                    for snake_index, move_state in record["opponent_states"]:

                        opponent_requests.append(
                            (record, snake_index, move_state)
                        )

                # 3. Group requests by opponent ID.
                opponent_groups = {}

                for record, snake_index, move_state in opponent_requests:

                    opponent_id = record["opponent_id"]

                    if opponent_id not in opponent_groups:
                        opponent_groups[opponent_id] = []

                    opponent_groups[opponent_id].append(
                        (record, snake_index, move_state)
                    )

                # 4. Request opponent actions from the main process.
                for opponent_id, requests in opponent_groups.items():

                    game_states = [
                        request[2] for request in requests
                    ]

                    moves = self.inference_client.select_actions(
                        opponent_id,
                        game_states
                    )

                    for (record, snake_index, _), move in zip(requests, moves):
                        if move is None:
                            record["state"]["board"]["snakes"][snake_index]["health"] = 0
                        else:
                            record["moves"][snake_index] = move

                # 5. Finish all prepared expansions.
                for record in expansion_records:
                    MCTS.finish_expansion(record)

                # 6. Select a child from each expanded node.
                for i, node in enumerate(selected_nodes):

                    if not node.done and len(node.children) > 0:

                        selected_nodes[i] = self.envs[i].mcts_rng.choice(
                            list(node.children.values())
                        )

                values = MCTS.evaluate_batch(
                    selected_nodes,
                    self.inference_client
                )

                # 8. Backpropagate.
                for node, value in zip(selected_nodes, values):
                    MCTS.backpropagate(node, value)

        finally:
            pass

        # Select the final action from each root.
        actions = []

        for env, root in zip(self.envs, roots):

            rng = env.mcts_rng

            if len(root.children) == 0:

                valid_actions = np.flatnonzero(root.mask)

                if len(valid_actions) == 0:
                    best_action = 1
                else:
                    best_action = int(
                        rng.choice(valid_actions.tolist())
                    )

            else:

                best_action = int(max(
                    root.children.items(),
                    key=lambda pair: pair[1].visit_count
                )[0])

            if env.flip:
                best_action = int(FLIP[best_action])

            actions.append(best_action)

        return actions

    #To compute the action mask for each environment:
    def get_masks(self):
        masks = [env.get_action_mask() for env in self.envs]
        return np.stack(masks)

#The following is a class for creating environment objects.  It includes functions for creating state observations, computing the agent's actions and simulating turns, canonicalizing and flipping the board, initializing games, and calculating rewards.
class Environment:
    def __init__(self, opponent_manager, device, width=11, height=11, num_opponents=3, seed=None):
        self.width = width
        self.height = height
        self.num_opponents = num_opponents

        self.opponent_manager = opponent_manager
        self.opponent_id = None
        self.device = torch.device("cpu")
        self.rng = random.Random(seed)
        self.mcts_rng = random.Random(None if seed is None else seed + 1)

        self.state = None
        self.agent_index = 0
        self.flip = False

    #To reset the environment by initializing a new game, choosing an opponent policy for the game, and returning the observation and action mask for the first game state:
    def reset(self):
        self.opponent_id = (
            self.opponent_manager.choose_opponent_id(self.rng)
        )
        self.state = GameSimulator.init_game(self.num_opponents + 1, 
                                              width=self.width,
                                              height=self.height, 
                                              rng=self.rng)
        
        self.state["you"] = self.state["board"]["snakes"][self.agent_index]
        self.flip = (self.rng.random() < 0.5)
        
        obs = self._get_observation()
        obs = self._apply_flip(obs)

        mask = self.get_action_mask()
        
        return obs, mask

    def prepare_step(self, action_index):
        """Prepare the agent move and collect opponent states."""

        action_index = int(action_index)

        if action_index not in (0, 1, 2):
            raise ValueError(
                f"Invalid action index: {action_index}"
            )

        if self.flip:
            action_index = int(FLIP[action_index])

        current_dir = self.get_current_direction()

        mapping = {
            "up": ["left", "up", "right"],
            "down": ["right", "down", "left"],
            "left": ["down", "left", "up"],
            "right": ["up", "right", "down"]
        }

        moves = {
            self.agent_index: mapping[current_dir][action_index]
        }

        opponent_states = []

        for i, snake in enumerate(self.state["board"]["snakes"]):
            if i == self.agent_index or snake["health"] <= 0:
                continue

            move_state = GameSimulator.make_move_state(
                self.state, i
            )

            opponent_states.append((i, move_state))

        return moves, opponent_states

    def finish_step(self, moves):
        """Resolve a prepared turn and return its transition."""

        GameSimulator.resolve_turn(self.state, moves, self.rng)

        self.state["you"] = self.state["board"]["snakes"][self.agent_index]
        self.state["turn"] += 1

        game_done = GameSimulator.game_over(self.state)
        agent_dead = (
            self.state["board"]["snakes"][self.agent_index]["health"] <= 0
        )

        done = agent_dead or game_done
        reward = self._compute_reward(game_done)

        self.flip = self.rng.random() < 0.5
        obs = self._apply_flip(self._get_observation())
        mask = self.get_action_mask()

        return obs, mask, reward, done, {}
    
    #To simulate a game turn, given the agent's move, and return the observation, mask, and reward for the resulting state:
    def step(self, action_index):
        moves, opponent_states = self.prepare_step(action_index)

        for snake_index, move_state in opponent_states:
            move = Policy.select_action(
                game_state=move_state,
                network=self.opponent_net,
                device=self.device
            )

            if move is None:
                self.state["board"]["snakes"][snake_index]["health"] = 0
            else:
                moves[snake_index] = move

        return self.finish_step(moves) 
    
    #The following function returns a reward of 1 if the agent is the last surviving snake at the end of a game, -1 if the agent's health is 0, and 0 otherwise. 
    def _compute_reward(self, done):
        agent = self.state["board"]["snakes"][self.agent_index]

        if agent["health"] <= 0:
            return -1.0

        if done: 
            return +1.0

        return 0.0
    
    #To create the obervation using a function from Policy.py:
    def _get_observation(self):
        return Policy._get_observation(self.state)

    #To calculate the agent's direction, given its head and body coordinates:
    def get_current_direction(self):
        snake = self.state["board"]["snakes"][self.agent_index]
        head = snake["head"]
        body = snake["body"]

        if len(body) <= 1:
            return "up"

        neck = body[1]
        if head["x"] > neck["x"]: return "right"
        elif head["x"] < neck["x"]: return "left"
        elif head["y"] > neck["y"]: return "up"
        elif head["y"] < neck["y"]: return "down"
        return "up"
       
    #The following function returns a list of three numbers representing each possible move.  If a move is either outside of the board's boundaries or results in a collision with a snake's body, including that of the agent, the number at that move's index is 0.  Otherwise, the number is 1.  This action mask is applied to Q-values to help the network train on "safe" moves:
    def get_action_mask(self):
        mask = np.ones(3, dtype=np.float32)
        snake = self.state["board"]["snakes"][self.agent_index]
        head = snake["head"]
        current_dir = self.get_current_direction()

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
            for snake in self.state["board"]["snakes"]
            if snake["health"] > 0
            for segment in snake["body"][:-1]
        }

        for i, move in enumerate(checks):
            if move["x"] < 0 or move["x"] >= self.width or move["y"] < 0 or move["y"] >= self.height:
                mask[i] = 0.0
                continue 

            if (move["x"], move["y"]) in occupied:
                mask[i] = 0.0

        if self.flip:
            mask = mask[FLIP]

        return mask

    #To reflect the observation on its vertical axis if the "flip" variable is set to True:
    def _apply_flip(self, grid):
        if self.flip:
            grid = np.flip(grid, axis=2).copy()
        return grid



