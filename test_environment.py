"""
Environment test script - Oyun ortamını test et
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.grid_world import TreasureHuntEnv
import numpy as np


def test_basic_functionality():
    """Temel environment fonksiyonlarını test et"""
    print("=" * 60)
    print("Testing Basic Environment Functionality")
    print("=" * 60)
    
    env = TreasureHuntEnv(grid_size=5, max_steps=50)
    
    # Reset test
    print("\n1. Testing reset()...")
    obs, info = env.reset()
    assert 'agent_pos' in obs
    assert 'treasure_pos' in obs
    assert 'grid' in obs
    print("   ✓ Reset successful")
    
    # Step test
    print("\n2. Testing step()...")
    action = 0  # north
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    print(f"   ✓ Step successful (reward: {reward:.2f})")
    
    # Render test
    print("\n3. Testing render()...")
    env.render()
    print("   ✓ Render successful")
    
    # Text description test
    print("\n4. Testing get_text_description()...")
    desc = env.get_text_description()
    print(f"   Description: {desc[:80]}...")
    assert isinstance(desc, str)
    assert len(desc) > 0
    print("   ✓ Text description successful")
    
    print("\n✅ All basic tests passed!")


def test_boundary_conditions():
    """Boundary condition'ları test et"""
    print("\n" + "=" * 60)
    print("Testing Boundary Conditions")
    print("=" * 60)
    
    env = TreasureHuntEnv(grid_size=5, max_steps=50)
    obs, info = env.reset()
    
    # Test wall collision
    print("\n1. Testing wall collision...")
    env.agent_pos = np.array([0, 0])
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward < 0
    print(f"   ✓ Wall collision handled (reward: {reward:.2f})")
    
    # Test treasure finding
    print("\n2. Testing treasure finding...")
    env.agent_pos = env.treasure_pos - np.array([1, 0])
    obs, reward, terminated, truncated, info = env.step(1)
    if np.array_equal(env.agent_pos, env.treasure_pos):
        assert reward > 0
        assert terminated
        print(f"   ✓ Treasure finding works (reward: {reward:.2f})")
    
    print("\n✅ All boundary tests passed!")


def test_action_space():
    """Action space'i test et"""
    print("\n" + "=" * 60)
    print("Testing Action Space")
    print("=" * 60)
    
    env = TreasureHuntEnv(grid_size=5, max_steps=50)
    obs, info = env.reset()
    
    print("\n1. Testing all actions...")
    initial_pos = env.agent_pos.copy()
    
    for action, name in enumerate(env.action_names):
        env.agent_pos = initial_pos.copy()
        env.grid[env.agent_pos[0], env.agent_pos[1]] = 2
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Action {action} ({name}): Agent moved to {tuple(env.agent_pos)}")
    
    print("\n✅ Action space test completed!")


def test_random_gameplay():
    """Random agent ile birkaç episode test et"""
    print("\n" + "=" * 60)
    print("Testing Random Gameplay")
    print("=" * 60)
    
    env = TreasureHuntEnv(grid_size=5, max_steps=50)
    
    num_episodes = 5
    successes = 0
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        obs, info = env.reset()
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = np.random.randint(0, 4)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            if terminated:
                successes += 1
        
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Steps: {steps}")
        print(f"   Success: {'✓' if terminated else '✗'}")
    
    print(f"\n📊 Success rate: {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")
    print("\n✅ Random gameplay test completed!")


def main():
    """Run all tests"""
    print("\n" + "🧪 " * 20)
    print("TREASURE HUNT ENVIRONMENT TEST SUITE")
    print("🧪 " * 20)
    
    try:
        test_basic_functionality()
        test_boundary_conditions()
        test_action_space()
        test_random_gameplay()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Environment is working correctly.")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())