import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TreasureHuntEnv(gym.Env):
    """
    A simple grid-based treasure hunt game. 
    The agent tries to find the treasure and avoid obstacles.
    """
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(self, grid_size=5, max_steps=50):
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: agent position + treasure position + obstacles
        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32),
            'treasure_pos': spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32),
            'grid': spaces.Box(low=0, high=3, shape=(grid_size, grid_size), dtype=np.int32)
        })
        
        self.action_to_direction = {
            0: np.array([-1, 0]),  # up
            1: np.array([1, 0]),   # down
            2: np.array([0, -1]),  # left
            3: np.array([0, 1])    # right
        }
        
        self.action_names = ['north', 'south', 'west', 'east']
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Initialize grid: 0=empty, 1=obstacle, 2=agent, 3=treasure
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Place agent at random position
        self.agent_pos = self._get_random_empty_pos()
        
        # Place treasure at random position (far from agent)
        while True:
            self.treasure_pos = self._get_random_empty_pos()
            if np.linalg.norm(self.agent_pos - self.treasure_pos) >= self.grid_size * 0.5:
                break
        
        # Place random obstacles
        num_obstacles = self.grid_size // 2
        for _ in range(num_obstacles):
            obs_pos = self._get_random_empty_pos()
            self.grid[obs_pos[0], obs_pos[1]] = 1
        
        # Update grid with agent and treasure
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 2
        self.grid[self.treasure_pos[0], self.treasure_pos[1]] = 3
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def _get_random_empty_pos(self):
        while True:
            pos = np.array([
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            ])
            if self.grid[pos[0], pos[1]] == 0:
                return pos
    
    def _get_obs(self):
        return {
            'agent_pos': self.agent_pos.copy(),
            'treasure_pos': self.treasure_pos.copy(),
            'grid': self.grid.copy()
        }
    
    def _get_info(self):
        distance = np.linalg.norm(self.agent_pos - self.treasure_pos)
        return {
            'distance_to_treasure': distance,
            'steps': self.current_step
        }
    
    def step(self, action):
        self.current_step += 1
        
        # Calculate new position
        direction = self.action_to_direction[action]
        new_pos = self.agent_pos + direction
        
        # Check boundaries
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            reward = -0.5  # Hit wall
            terminated = False
            truncated = self.current_step >= self.max_steps
            return self._get_obs(), reward, terminated, truncated, self._get_info()
        
        # Check obstacle
        if self.grid[new_pos[0], new_pos[1]] == 1:
            reward = -1.0  # Hit obstacle
            terminated = False
            truncated = self.current_step >= self.max_steps
            return self._get_obs(), reward, terminated, truncated, self._get_info()
        
        # Update grid
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 0
        self.agent_pos = new_pos
        
        # Check if found treasure
        if np.array_equal(self.agent_pos, self.treasure_pos):
            reward = 10.0  # Found treasure!
            terminated = True
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 2
            return self._get_obs(), reward, terminated, False, self._get_info()
        
        # Small negative reward for each step (encourage efficiency)
        distance = np.linalg.norm(self.agent_pos - self.treasure_pos)
        reward = -0.1 - 0.05 * distance  # Distance-based penalty
        
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 2
        
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self):
        """Human-readable text representation"""
        symbols = {0: '·', 1: '█', 2: 'A', 3: 'T'}
        
        print(f"\n=== Step {self.current_step} ===")
        for i in range(self.grid_size):
            row = ''
            for j in range(self.grid_size):
                row += symbols[self.grid[i, j]] + ' '
            print(row)
        print(f"Distance to treasure: {np.linalg.norm(self.agent_pos - self.treasure_pos):.2f}\n")
    
    def get_text_description(self):
        """LLM için metin açıklaması"""
        distance = np.linalg.norm(self.agent_pos - self.treasure_pos)
        
        # Check surroundings
        surroundings = []
        for action, direction in self.action_to_direction.items():
            check_pos = self.agent_pos + direction
            if 0 <= check_pos[0] < self.grid_size and 0 <= check_pos[1] < self.grid_size:
                cell = self.grid[check_pos[0], check_pos[1]]
                if cell == 1:
                    surroundings.append(f"obstacle to the {self.action_names[action]}")
                elif cell == 3:
                    surroundings.append(f"TREASURE to the {self.action_names[action]}")
        
        description = f"You are at position {tuple(self.agent_pos)}. "
        description += f"The treasure is at position {tuple(self.treasure_pos)}. "
        description += f"Distance to treasure: {distance:.1f} steps. "
        
        if surroundings:
            description += f"You see: {', '.join(surroundings)}. "
        
        description += "Available actions: north, south, west, east."
        
        return description