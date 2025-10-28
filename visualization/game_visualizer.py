"""
Grid world visualization with Pygame
"""
import pygame
import sys
import os
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.grid_world import TreasureHuntEnv
from agent.llm_agent import LLMAgent

# Renkler
COLORS = {
    'background': (20, 20, 30),
    'grid': (40, 40, 50),
    'empty': (50, 50, 60),
    'obstacle': (80, 30, 30),
    'agent': (50, 150, 250),
    'treasure': (255, 215, 0),
    'text': (255, 255, 255),
    'path': (100, 100, 255, 100)
}


class GameVisualizer:
    def __init__(self, env, agent=None, cell_size=80):
        pygame.init()

        self.env = env
        self.agent = agent
        self.cell_size = cell_size
        self.grid_size = env.grid_size

        # Window boyutları
        self.grid_width = self.grid_size * cell_size
        self.sidebar_width = 300
        self.window_width = self.grid_width + self.sidebar_width
        self.window_height = self.grid_size * cell_size

        # Ekran oluştur
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("🎮 RL Agent - Treasure Hunt")

        # Font
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 24)

        # Clock
        self.clock = pygame.time.Clock()

        # Game state
        self.history = []
        self.total_reward = 0
        self.steps = 0
        self.episode = 0
        self.mode = "AI"  # "AI" or "HUMAN"

        # Animation
        self.agent_pos_visual = None

    def draw_grid(self):
        """Grid çiz"""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = j * self.cell_size
                y = i * self.cell_size

                # Cell type'a göre renk seç
                cell_value = self.env.grid[i, j]

                if cell_value == 0:  # Empty
                    color = COLORS['empty']
                elif cell_value == 1:  # Obstacle
                    color = COLORS['obstacle']
                elif cell_value == 2:  # Agent
                    color = COLORS['agent']
                elif cell_value == 3:  # Treasure
                    color = COLORS['treasure']
                else:
                    color = COLORS['empty']

                # Kare çiz
                pygame.draw.rect(self.screen, color, (x, y, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, COLORS['grid'], (x, y, self.cell_size, self.cell_size), 2)

                # Icons ekle
                if cell_value == 1:  # Obstacle
                    self.draw_text("🧱", x + self.cell_size // 2, y + self.cell_size // 2, self.font_large)
                elif cell_value == 2:  # Agent
                    self.draw_text("🤖", x + self.cell_size // 2, y + self.cell_size // 2, self.font_large)
                elif cell_value == 3:  # Treasure
                    self.draw_text("💎", x + self.cell_size // 2, y + self.cell_size // 2, self.font_large)

    def draw_sidebar(self):
        """Sağ tarafta bilgi paneli"""
        x_start = self.grid_width + 20
        y = 20

        # Başlık
        self.draw_text("📊 Game Stats", x_start, y, self.font_large, COLORS['text'], align='left')
        y += 50

        # Episode
        self.draw_text(f"Episode: {self.episode}", x_start, y, self.font_medium, COLORS['text'], align='left')
        y += 35

        # Steps
        self.draw_text(f"Steps: {self.steps}", x_start, y, self.font_medium, COLORS['text'], align='left')
        y += 35

        # Reward
        color = (100, 255, 100) if self.total_reward > 0 else (255, 100, 100)
        self.draw_text(f"Reward: {self.total_reward:.2f}", x_start, y, self.font_medium, color, align='left')
        y += 35

        # Distance
        distance = np.linalg.norm(self.env.agent_pos - self.env.treasure_pos)
        self.draw_text(f"Distance: {distance:.1f}", x_start, y, self.font_medium, COLORS['text'], align='left')
        y += 50

        # Mode
        self.draw_text(f"Mode: {self.mode}", x_start, y, self.font_medium, (255, 215, 0), align='left')
        y += 50

        # Controls
        self.draw_text("⌨️ Controls:", x_start, y, self.font_medium, COLORS['text'], align='left')
        y += 35

        controls = [
            "↑ ↓ ← → : Move",
            "SPACE : AI Step",
            "A : Auto Play",
            "R : Reset",
            "ESC : Quit"
        ]

        for control in controls:
            self.draw_text(control, x_start, y, self.font_small, (180, 180, 180), align='left')
            y += 25

        # Son aksiyon
        if self.history:
            y += 20
            self.draw_text("Last Action:", x_start, y, self.font_small, COLORS['text'], align='left')
            y += 25
            last = self.history[-1]
            self.draw_text(last[:25] + "...", x_start, y, self.font_small, (150, 150, 255), align='left')

    def draw_text(self, text, x, y, font, color=None, align='center'):
        """Metin çiz"""
        if color is None:
            color = COLORS['text']

        text_surface = font.render(str(text), True, color)
        text_rect = text_surface.get_rect()

        if align == 'center':
            text_rect.center = (x, y)
        elif align == 'left':
            text_rect.topleft = (x, y)

        self.screen.blit(text_surface, text_rect)

    def handle_input(self):
        """Kullanıcı input'u işle"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "QUIT"

                elif event.key == pygame.K_r:
                    return "RESET"

                elif event.key == pygame.K_SPACE and self.agent:
                    return "AI_STEP"

                elif event.key == pygame.K_a and self.agent:
                    return "AUTO_PLAY"

                # Manuel hareket
                elif event.key == pygame.K_UP:
                    return "MOVE", 0
                elif event.key == pygame.K_DOWN:
                    return "MOVE", 1
                elif event.key == pygame.K_LEFT:
                    return "MOVE", 2
                elif event.key == pygame.K_RIGHT:
                    return "MOVE", 3

        return None

    def step(self, action):
        """Bir adım at"""
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.total_reward += reward
        self.steps += 1
        self.history.append(f"{self.env.action_names[action]} → {reward:.2f}")

        return obs, reward, terminated, truncated, info

    def reset(self):
        """Oyunu sıfırla"""
        self.env.reset()
        self.history = []
        self.total_reward = 0
        self.steps = 0
        self.episode += 1

    def run(self, fps=5, auto_play=False):
        """Ana oyun döngüsü"""
        running = True
        auto = auto_play

        while running:
            self.clock.tick(fps)

            # Input handle
            action_result = self.handle_input()

            if action_result == "QUIT":
                running = False
                break

            elif action_result == "RESET":
                self.reset()
                auto = False

            elif action_result == "AUTO_PLAY":
                auto = not auto

            elif action_result == "AI_STEP" and self.agent:
                # AI bir adım atsın
                state_desc = self.env.get_text_description()
                action, response = self.agent.get_action(state_desc, self.history)
                obs, reward, terminated, truncated, info = self.step(action)

                if terminated or truncated:
                    pygame.time.wait(1000)
                    self.reset()

            elif action_result and action_result[0] == "MOVE":
                # Manuel hareket
                self.mode = "HUMAN"
                action = action_result[1]
                obs, reward, terminated, truncated, info = self.step(action)

                if terminated or truncated:
                    pygame.time.wait(1000)
                    self.reset()

            # Auto play mode
            if auto and self.agent:
                self.mode = "AI"
                state_desc = self.env.get_text_description()
                action, response = self.agent.get_action(state_desc, self.history)
                obs, reward, terminated, truncated, info = self.step(action)

                if terminated or truncated:
                    pygame.time.wait(1000)
                    self.reset()

            # Çiz
            self.screen.fill(COLORS['background'])
            self.draw_grid()
            self.draw_sidebar()

            pygame.display.flip()

        pygame.quit()


def main():
    """Ana fonksiyon"""
    import argparse

    parser = argparse.ArgumentParser(description="Pygame Visualization")
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint')
    parser.add_argument('--grid-size', type=int, default=5, help='Grid size')
    parser.add_argument('--cell-size', type=int, default=80, help='Cell size in pixels')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second')
    parser.add_argument('--auto', action='store_true', help='Auto play mode')

    args = parser.parse_args()

    # Environment
    env = TreasureHuntEnv(grid_size=args.grid_size)

    # Agent
    agent = None
    if args.checkpoint:
        print(f"Loading agent from {args.checkpoint}")
        agent = LLMAgent()
        agent.load(args.checkpoint)

    # Visualizer
    visualizer = GameVisualizer(env, agent, cell_size=args.cell_size)

    # Run
    visualizer.run(fps=args.fps, auto_play=args.auto)


if __name__ == "__main__":
    main()