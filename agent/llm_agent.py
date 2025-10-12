import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from torch.nn import functional as F
import re

class LLMAgent:
    """
    LLM-based game agent.
    Uses a model like GPT-2 for text-based decision making.
    """
    
    def __init__(
        self, 
        model_name="distilgpt2",
        device="cuda" if torch.cuda.is_available() else "cpu",
        temperature=0.7,
        max_length=100
    ):
        self.device = device
        self.temperature = temperature
        self.max_length = max_length
        
        print(f"Loading model: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        # Add padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        self.action_keywords = ['north', 'south', 'west', 'east']
        self.action_map = {
            'north': 0, 'up': 0,
            'south': 1, 'down': 1,
            'west': 2, 'left': 2,
            'east': 3, 'right': 3
        }
    
    def create_prompt(self, state_description, history=None):
        """Environment state'inden prompt oluştur"""
        prompt = "You are playing a treasure hunt game. "
        prompt += state_description + "\n"
        
        if history:
            prompt += "\nPrevious actions:\n"
            for h in history[-3:]:  # Son 3 aksiyon
                prompt += f"- {h}\n"
        
        prompt += "\nWhat action do you take? Choose one: north, south, west, east.\nAction:"
        
        return prompt
    
    def parse_action(self, text):
        """LLM output'undan action çıkar"""
        text = text.lower()
        
        # Direkt keyword ara
        for keyword, action_id in self.action_map.items():
            if keyword in text:
                return action_id
        
        # Varsayılan: random action
        return np.random.randint(0, 4)
    
    def get_action(self, state_description, history=None, deterministic=False):
        """
        State description'dan action üret
        """
        prompt = self.create_prompt(state_description, history)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        action = self.parse_action(response)
        
        return action, response
    
    def compute_loss_with_reward(self, prompts, actions, rewards):
        """
        RL için custom loss: reward-weighted cross-entropy
        """
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        outputs = self.model(**inputs, labels=inputs['input_ids'])
        
        # Base loss
        base_loss = outputs.loss
        
        # Reward weighting
        reward_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        reward_weight = torch.sigmoid(reward_tensor)  # Normalize rewards
        
        # Weighted loss
        weighted_loss = base_loss * reward_weight.mean()
        
        return weighted_loss
    
    def save(self, path):
        """Model'i kaydet"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Model'i yükle"""
        self.model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"Model loaded from {path}")


class RewardBuffer:
    """
    Experience replay buffer for RL training
    """
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = []
    
    def add(self, state_desc, action, reward, next_state_desc, done):
        experience = {
            'state': state_desc,
            'action': action,
            'reward': reward,
            'next_state': next_state_desc,
            'done': done
        }
        
        self.buffer.append(experience)
        
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


def train_step(agent, batch, optimizer):
    """
    Tek bir training step
    """
    prompts = []
    rewards = []
    
    for exp in batch:
        prompt = agent.create_prompt(exp['state'])
        prompts.append(prompt)
        rewards.append(exp['reward'])
    
    optimizer.zero_grad()
    loss = agent.compute_loss_with_reward(prompts, None, rewards)
    loss.backward()
    optimizer.step()
    
    return loss.item()