import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from environment.grid_world import TreasureHuntEnv
from agent.llm_agent import LLMAgent, RewardBuffer, train_step


class RLTrainer:
    """
    LLM + RL Training Pipeline
    """
    def __init__(
        self,
        agent,
        env,
        buffer_size=1000,
        batch_size=16,
        learning_rate=5e-5,
        save_dir="checkpoints"
    ):
        self.agent = agent
        self.env = env
        self.buffer = RewardBuffer(buffer_size)
        self.batch_size = batch_size
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            agent.model.parameters(),
            lr=learning_rate
        )
        
        # Metrics
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': [],
            'losses': []
        }
    
    def collect_episode(self, render=False):
        """
        Bir episode oyna ve experience topla
        """
        obs, info = self.env.reset()
        state_desc = self.env.get_text_description()
        
        episode_reward = 0
        episode_length = 0
        done = False
        history = []
        
        while not done:
            # Agent action seç
            action, response = self.agent.get_action(state_desc, history)
            history.append(f"{response} -> {self.env.action_names[action]}")
            
            # Environment step
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            next_state_desc = self.env.get_text_description()
            
            # Buffer'a ekle
            self.buffer.add(state_desc, action, reward, next_state_desc, done)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                self.env.render()
                print(f"Action: {response} -> {self.env.action_names[action]}")
                print(f"Reward: {reward:.2f}")
            
            state_desc = next_state_desc
        
        return episode_reward, episode_length, terminated
    
    def train(
        self, 
        num_episodes=100, 
        train_freq=5,
        eval_freq=10,
        save_freq=20
    ):
        """
        Ana training loop
        """
        print("=" * 60)
        print("Starting LLM + RL Training")
        print("=" * 60)
        
        progress_bar = tqdm(range(num_episodes), desc="Training")
        
        for episode in progress_bar:
            # Episode topla
            ep_reward, ep_length, success = self.collect_episode(render=False)
            
            self.metrics['episode_rewards'].append(ep_reward)
            self.metrics['episode_lengths'].append(ep_length)
            
            # Training step
            if len(self.buffer) >= self.batch_size and episode % train_freq == 0:
                batch = self.buffer.sample(self.batch_size)
                loss = train_step(self.agent, batch, self.optimizer)
                self.metrics['losses'].append(loss)
            else:
                loss = 0.0
            
            # Success rate hesapla
            recent_successes = sum(self.metrics['episode_rewards'][-10:]) > 0
            success_rate = recent_successes / min(10, episode + 1)
            self.metrics['success_rate'].append(success_rate)
            
            # Progress bar güncelle
            progress_bar.set_postfix({
                'reward': f"{ep_reward:.2f}",
                'length': ep_length,
                'success': success,
                'loss': f"{loss:.4f}",
                'buffer': len(self.buffer)
            })
            
            # Evaluation
            if episode % eval_freq == 0 and episode > 0:
                eval_reward, eval_length, eval_success = self.evaluate(num_episodes=5)
                print(f"\n[Eval Episode {episode}] Avg Reward: {eval_reward:.2f}, "
                      f"Avg Length: {eval_length:.1f}, Success Rate: {eval_success*100:.1f}%")
            
            # Save checkpoint
            if episode % save_freq == 0 and episode > 0:
                self.save_checkpoint(episode)
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        
        self.save_metrics()
    
    def evaluate(self, num_episodes=10):
        """
        Agent'i evaluate et
        """
        rewards = []
        lengths = []
        successes = []
        
        for _ in range(num_episodes):
            ep_reward, ep_length, success = self.collect_episode(render=False)
            rewards.append(ep_reward)
            lengths.append(ep_length)
            successes.append(1 if success else 0)
        
        return np.mean(rewards), np.mean(lengths), np.mean(successes)
    
    def save_checkpoint(self, episode):
        """
        Model checkpoint kaydet
        """
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_episode_{episode}")
        self.agent.save(checkpoint_path)
        print(f"\nCheckpoint saved at episode {episode}")
    
    def save_metrics(self):
        """
        Training metrics'i kaydet
        """
        metrics_path = os.path.join(self.save_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")


def main():
    """
    Ana training fonksiyonu
    """
    # Hyperparameters
    config = {
        'model_name': 'distilgpt2',
        'grid_size': 5,
        'max_steps': 50,
        'num_episodes': 200,
        'buffer_size': 1000,
        'batch_size': 16,
        'learning_rate': 5e-5,
        'train_freq': 5,
        'eval_freq': 20,
        'save_freq': 50,
        'temperature': 0.7
    }
    
    print("Configuration:")
    print(json.dumps(config, indent=2))
    print()
    
    # Environment oluştur
    env = TreasureHuntEnv(
        grid_size=config['grid_size'],
        max_steps=config['max_steps']
    )
    
    # Agent oluştur
    agent = LLMAgent(
        model_name=config['model_name'],
        temperature=config['temperature']
    )
    
    # Trainer oluştur
    trainer = RLTrainer(
        agent=agent,
        env=env,
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )
    
    # Training başlat
    trainer.train(
        num_episodes=config['num_episodes'],
        train_freq=config['train_freq'],
        eval_freq=config['eval_freq'],
        save_freq=config['save_freq']
    )
    
    # Final evaluation
    print("\nFinal Evaluation:")
    eval_reward, eval_length, eval_success = trainer.evaluate(num_episodes=20)
    print(f"Average Reward: {eval_reward:.2f}")
    print(f"Average Episode Length: {eval_length:.1f}")
    print(f"Success Rate: {eval_success*100:.1f}%")


if __name__ == "__main__":
    main()