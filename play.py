import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from environment.grid_world import TreasureHuntEnv
from agent.llm_agent import LLMAgent
try:
    import pygame
    from visualization.game_visualizer import GameVisualizer
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  Pygame not installed. Install with: pip install pygame")


def play_game(agent, env, num_episodes=5, render=True):
    """
    Agent ile oyun oyna ve sonuçları göster
    """
    results = []
    
    for episode in range(num_episodes):
        print("\n" + "=" * 60)
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print("=" * 60)
        
        obs, info = env.reset()
        state_desc = env.get_text_description()
        
        episode_reward = 0
        episode_length = 0
        done = False
        history = []
        
        if render:
            env.render()
        
        while not done:
            # Agent decision
            action, response = agent.get_action(state_desc, history, deterministic=True)
            action_name = env.action_names[action]
            
            history.append(f"{action_name}")
            
            if render:
                print(f"\n📍 Current State: {state_desc}")
                print(f"🤖 Agent thinking: '{response}'")
                print(f"➡️  Action: {action_name.upper()}")
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
                print(f"💰 Reward: {reward:.2f} | Total: {episode_reward:.2f}")
                
                if terminated:
                    print("\n🎉 TREASURE FOUND! SUCCESS!")
                elif truncated:
                    print("\n⏱️  TIME'S UP! Game over.")
            
            state_desc = env.get_text_description()
        
        results.append({
            'episode': episode + 1,
            'reward': episode_reward,
            'length': episode_length,
            'success': terminated
        })
        
        print(f"\n📊 Episode Summary:")
        print(f"   Total Reward: {episode_reward:.2f}")
        print(f"   Steps Taken: {episode_length}")
        print(f"   Status: {'✅ SUCCESS' if terminated else '❌ FAILED'}")
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    
    total_successes = sum(1 for r in results if r['success'])
    avg_reward = sum(r['reward'] for r in results) / len(results)
    avg_length = sum(r['length'] for r in results) / len(results)
    
    print(f"Success Rate: {total_successes}/{num_episodes} ({total_successes/num_episodes*100:.1f}%)")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.1f}")
    
    return results


def interactive_mode(agent, env):
    """
    Kullanıcı ile interactive oyun modu
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE - You control the agent!")
    print("=" * 60)
    print("Commands: north, south, west, east, auto, quit")
    print()
    
    obs, info = env.reset()
    env.render()
    
    done = False
    episode_reward = 0
    
    while not done:
        state_desc = env.get_text_description()
        print(f"\n📍 {state_desc}")
        
        # User input
        user_input = input("Your command: ").strip().lower()
        
        if user_input == 'quit':
            print("Thanks for playing!")
            break
        
        if user_input == 'auto':
            action, response = agent.get_action(state_desc, None, deterministic=True)
            action_name = env.action_names[action]
            print(f"🤖 Agent suggests: {action_name} (reasoning: '{response}')")
            confirm = input("Execute this action? (y/n): ").strip().lower()
            if confirm != 'y':
                continue
        else:
            # Parse user action
            action_map = {'north': 0, 'south': 1, 'west': 2, 'east': 3}
            if user_input not in action_map:
                print("❌ Invalid action! Use: north, south, west, east, auto, quit")
                continue
            action = action_map[user_input]
            action_name = user_input
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
        env.render()
        print(f"💰 Reward: {reward:.2f} | Total: {episode_reward:.2f}")
        
        if terminated:
            print("\n🎉 TREASURE FOUND! You won!")
        elif truncated:
            print("\n⏱️  TIME'S UP! Game over.")
    
    print(f"\nFinal Score: {episode_reward:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Play Treasure Hunt with LLM Agent")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: use base model)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to play')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode (you control the agent)')
    parser.add_argument('--visual', action='store_true',
                        help='Visual mode with Pygame')
    parser.add_argument('--text', action='store_true',
                        help='Force text mode')
    parser.add_argument('--grid-size', type=int, default=5,
                        help='Size of the grid')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    parser.add_argument('--model', type=str, default='distilgpt2',
                        help='Base model name')
    
    args = parser.parse_args()
    
    # Create environment
    env = TreasureHuntEnv(grid_size=args.grid_size)
    
    # Create agent
    agent = LLMAgent(model_name=args.model)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        agent.load(args.checkpoint)

        # Run appropriate mode

        if args.text:
            play_game(agent, env, num_episodes=args.episodes, render=not args.no_render)
        elif args.interactive:
            interactive_mode(agent, env)
        else:

            if PYGAME_AVAILABLE and not args.no_render:
                play_visual(agent, env, args.checkpoint)
            else:
                play_game(agent, env, num_episodes=args.episodes, render=not args.no_render)


def play_visual(agent, env, checkpoint_path=None):
    """
    Visual game mode with Pygame
    """
    if not PYGAME_AVAILABLE:
        print("❌ Pygame is not installed! To install:")
        print("   pip install pygame")
        return

    print("\n" + "=" * 60)
    print("🎮 Visual Game Mode - Pygame")
    print("=" * 60)
    print("\nControls:")
    print("  ⬆️ ⬇️ ⬅️ ➡️  : Move manually")
    print("  SPACE    : AI makes one move")
    print("  A        : Auto-play mode (AI continuous)")
    print("  R        : Reset game")
    print("  ESC      : Quit")
    print("\n" + "=" * 60)

    # Create Visualizer
    visualizer = GameVisualizer(env, agent, cell_size=80)

    # Start the Game
    visualizer.run(fps=5, auto_play=False)

if __name__ == "__main__":
    main()