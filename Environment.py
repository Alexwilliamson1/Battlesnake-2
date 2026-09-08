import random
import numpy as np
import GameSimulator
import MCTS
import Policy

#An array of the action indices in reverse order:
FLIP = np.array([2, 1, 0])

#The following is a class for creating and calling functions on a vector of environment objects.  A VectorizedEnv object is initialized with a number of environments, an OpponentManager object, and a device, such as "cpu."
class VectorizedEnv:
    def __init__(self, num_envs, opponent_manager, device):
        self.envs = [Environment(opponent_manager=opponent_manager, device=device) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.games_complete = 0
        self.reward = 0

    #To reset each environment by 
    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, masks = zip(*results)
        return np.stack(obs), np.stack(masks)

    #To simulate turns, initialize new games, and return the observation, action mask, reward, and done indicator for the transition in each environment:
    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.envs, actions)]

        obs, masks, rewards, dones, infos = zip(*results)
        
        obs = list(obs)
        masks = list(masks)
        for i, done in enumerate(dones):
            if done:
                new_obs, new_mask = self.envs[i].reset()
                obs[i] = new_obs
                masks[i] = new_mask
                self.games_complete += 1
                self.reward += rewards[i]
                if self.games_complete % 20 == 0:
                    print(f"{self.games_complete} games have run.")
                    print("Average reward (20 games): ", self.reward / 20)
                    self.reward = 0
        return (
            np.stack(obs),
            np.stack(masks),
            np.array(rewards),
            np.array(dones),
            infos
        )

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
        self.opponent_net = None
        self.device = device
        self.rng = random.Random(seed)
        self.mcts_rng = random.Random(None if seed is None else seed + 1)

        self.state = None
        self.agent_index = 0
        self.flip = False

    #To reset the environment by initializing a new game, choosing an opponent policy for the game, and returning the observation and action mask for the first game state:
    def reset(self):
        self.opponent_net = (self.opponent_manager.choose_opponent(self.rng))

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

    #To simulate a game turn, given the agent's move, and return the observation, mask, and reward for the resulting state:
    def step(self, action_index):
        action_index  = int(action_index) 
        current_flip = self.flip

        if current_flip:
            action_index = int(FLIP[action_index])

        if action_index not in (0, 1, 2):
            raise ValueError(f"The action index must be 0, 1, or 2, but it is {action_index}")

        current_dir = self.get_current_direction()
        mapping = {
            "up": ["left", "up", "right"],
            "down": ["right", "down", "left"], 
            "left": ["down", "left", "up"],
            "right": ["up", "right", "down"]
        }

        moves = {self.agent_index: mapping[current_dir][action_index]}

        for i, snake in enumerate(self.state["board"]["snakes"]):
            if i == self.agent_index:
                continue
            if snake["health"] <= 0:
                continue

            move_state = GameSimulator.make_move_state(self.state, i)
            
            move = Policy.select_action(game_state=move_state, network=self.opponent_net, device=self.device) 

            if move is None:
                snake["health"] = 0
                continue

            moves[i] = move
            
        GameSimulator.resolve_turn(self.state, moves, self.rng)
        self.state["you"] = (self.state["board"]["snakes"][self.agent_index])
        self.state["turn"] += 1

        game_done = GameSimulator.game_over(self.state)

        agent_dead = (self.state["board"]["snakes"][self.agent_index]["health"] <= 0)

        done = agent_dead or game_done
        reward = self._compute_reward(game_done)

        self.flip = self.rng.random() < 0.5
        obs = self._apply_flip(self._get_observation())

        mask = self.get_action_mask()

        return obs, mask, reward, done, {}
   
    #The following function returns a reward of 1 if the agent is the last surviving snake at the end of a game, -1 if the agent's health is 0, and 0 otherwise. 
    def _compute_reward(self, done):
        agent = self.state["board"]["snakes"][self.agent_index]

        if agent["health"] <= 0:
            return -1.0

        if done: 
            return +1.0

        return 0.0

    #To perform a Monte Carlo Tree Search with the given number of simulations and return the move with most visit counts:
    def mcts_action(self, policy_net, num_simulations=10):
        rng = self.mcts_rng
        mask = MCTS.get_action_mask(self.state)
        
        root = MCTS.Node(self.state, mask, observation=self._get_observation(), rng=rng)

        for _ in range(num_simulations):
            node = MCTS.select(root)
            
            if node.done:
                value = MCTS.evaluate(node, policy_net, self.device)

            elif node.visit_count == 0 and node.parent is not None:
                value = MCTS.evaluate(node, policy_net, self.device)

            else:
                if len(node.children) == 0:
                    MCTS.expand(node, self.opponent_net, self.device)

                if len(node.children) > 0:
                    node = rng.choice(
                        list(node.children.values())
                    )
            
                value = MCTS.evaluate(node, policy_net, self.device)
            
            MCTS.backpropagate(node, value)
        
        if len(root.children) == 0:
            valid_actions = np.flatnonzero(mask)
            if len(valid_actions) == 0:
                best_action = 1
            else:    
                best_action = int(rng.choice(valid_actions.tolist()))
        else:
            best_action = int(max(
                root.children.items(),
                key = lambda pair: pair[1].visit_count
            )[0])
        
        if self.flip:
            best_action = int(FLIP[best_action])

        return best_action
    
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



