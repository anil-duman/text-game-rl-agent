# 🎮 Text-based Mini Game Agent (LLM + RL)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)


A reinforcement learning project that trains language models (LLMs) to play text-based games using reward shaping and experience replay. The agent learns to navigate a grid world, avoid obstacles, and find treasure through trial and error.

## Features

- **LLM-based Agent**: Uses DistilGPT-2/GPT-2 for natural language decision making
- **Reinforcement Learning**: Experience replay buffer with reward-weighted training
- **Grid World Environment**: Customizable treasure hunt game with obstacles
- **Training Metrics**: Real-time progress tracking and visualization
- **Interactive Mode**: Play alongside or against the trained agent
- **Model Checkpointing**: Save and load trained models
- **Docker Support**: Fully containerized for reproducibility

## Demo

```
=== Step 15 ===
· · · · ·
· █ · · T
· · A · ·
· █ · · ·
· · · · ·

Agent: "I should go east to get closer to the treasure"
Action: EAST → Reward: -0.2
Distance to treasure: 2.2 steps
```

## Quick Start

### Prerequisites

- Python 3.8+
- pip
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/anil-duman/text-game-rl-agent.git
   cd text-game-rl-agent
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Test the environment**
   ```bash
   python test_environment.py
   ```

### Usage

#### Train the Agent

```bash
python train.py
```

**Training output:**
```
Configuration:
{
  "model_name": "distilgpt2",
  "num_episodes": 200,
  "grid_size": 5,
  ...
}

Training:  45%|████▌     | 90/200 [12:34<15:23, 8.39s/it, 
    reward=4.2, length=28, success=True, loss=0.0234]

[Eval Episode 100] Avg Reward: 4.82, Success Rate: 55.0%
```

#### Play with Trained Agent

```bash
# Watch agent play 5 episodes
python play.py --checkpoint checkpoints/checkpoint_episode_100 --episodes 5

# Interactive mode (you control the agent)
python play.py --interactive

# Play with base model (no training)
python play.py --episodes 3
```

## How It Works

### 1. Environment

The **Grid World** is a 5×5 grid where:
- **Agent (A)**: Controlled by the LLM
- **Treasure (T)**: Goal to reach
- **Obstacles (█)**: Walls to avoid
- **Empty space**: Free to move

### 2. LLM Decision Making

The agent receives text descriptions:
```
You are at position (2, 1). The treasure is at position (4, 4).
Distance to treasure: 3.6 steps. You see: obstacle to the east.
Available actions: north, south, west, east.
```

The LLM generates natural language responses:
```
Agent: "I should go south to avoid the obstacle and get closer"
Action: SOUTH
```

### 3. Reward Shaping

```python
# Treasure found
+10.0

# Hit obstacle
-1.0

# Hit wall
-0.5

# Regular step
-0.1 - 0.05 × distance_to_treasure
```

### 4. Training Process

```
┌─────────────────────┐
│  Collect Episode    │
│  (Agent plays game) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Store Experience   │
│  (Replay Buffer)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Sample Batch       │
│  Train with Rewards │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Update Model       │
│  (Gradient Descent) │
└─────────────────────┘
```

## Expected Results

After 200 episodes:

| Metric | Random Baseline | Trained Agent |
|--------|----------------|---------------|
| Success Rate | ~10% | **40-60%** |
| Average Reward | -2.0 | **3-5** |
| Episode Length | 45-50 | **25-35** |

## Configuration

Edit `train.py` to customize:

```python
config = {
    'model_name': 'distilgpt2',     # or 'gpt2', 'gpt2-medium'
    'grid_size': 5,                 # 5x5 grid
    'max_steps': 50,                # max steps per episode
    'num_episodes': 200,            # total training episodes
    'batch_size': 16,               # training batch size
    'learning_rate': 5e-5,          # AdamW learning rate
    'temperature': 0.7,             # sampling temperature (0.0-1.0)
    'train_freq': 5,                # train every N episodes
}
```

## Project Structure

```
text-game-rl-agent/
├── environment/
│   ├── __init__.py
│   └── grid_world.py          # Game environment
├── agent/
│   ├── __init__.py
│   ├── llm_agent.py           # LLM agent + RL training
│   └── ppo_trainer.py         # Advanced RL (optional)
├── checkpoints/               # Saved models
├── logs/                      # Training logs
├── train.py                   # Training script
├── play.py                    # Play/evaluation script
├── test_environment.py        # Test suite
├── requirements.txt           # Dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Advanced Usage

### Custom Environment

```python
from environment.grid_world import TreasureHuntEnv

# Larger grid
env = TreasureHuntEnv(grid_size=7, max_steps=100)

# Custom reward function (edit grid_world.py)
reward = -0.1 - 0.1 * distance  # Stronger distance penalty
```

### Different Models

```python
from agent.llm_agent import LLMAgent

# Larger model (requires more memory)
agent = LLMAgent(model_name='gpt2')

# More deterministic
agent = LLMAgent(temperature=0.5)

# CPU only
agent = LLMAgent(device='cpu')
```

### Training Tips

**Faster convergence:**
- Start with smaller grid: `grid_size=3`
- Increase treasure reward: `+20` instead of `+10`
- Lower temperature: `0.5-0.6`

**Better exploration:**
- Higher temperature: `0.8-0.9`
- Larger replay buffer: `buffer_size=2000`

**More stable:**
- Lower learning rate: `1e-5`
- Smaller batch size: `batch_size=8`

## Troubleshooting

### Out of Memory Error
```bash
# Use smaller model
agent = LLMAgent(model_name='distilgpt2')

# Reduce batch size
batch_size = 8

# Use CPU
agent = LLMAgent(device='cpu')
```

### Slow Training
```bash
# Smaller grid
grid_size = 3

# Fewer episodes
num_episodes = 50

# Train less frequently
train_freq = 10
```

### Agent Not Learning
- Train for more episodes (500+)
- Adjust reward shaping
- Try different temperature (0.5-0.7)
- Check if model is updating (monitor loss)

## Dependencies

- **torch** - Deep learning framework
- **transformers** - Hugging Face LLM library
- **numpy** - Numerical computing
- **gymnasium** - RL environment interface
- **tqdm** - Progress bars

See `requirements.txt` for full list.

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Implement PPO (Proximal Policy Optimization)
- [ ] Add more environment types (maze, RPG)
- [ ] Web interface for playing
- [ ] Curriculum learning
- [ ] Multi-agent support
- [ ] Better visualization

## Acknowledgments

- OpenAI Gymnasium for environment interface
- Hugging Face for transformer models
- PyTorch team for the deep learning framework

## Author

- Anil Duman
