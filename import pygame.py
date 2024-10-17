import pygame
import random
import numpy as np
import sys
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH = 600
HEIGHT = 600
CELL_SIZE = 20

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN = (0, 180, 0)
DARK_GREEN = (0, 100, 0)
LIGHT_GREEN = (144, 238, 144)
GRAY = (50, 50, 50)
TRANSLUCENT_GREEN = (0, 255, 0, 50)  # Translucent green for simulations

# Directions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Create the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Monte Carlo Tree Snake')

clock = pygame.time.Clock()

# Fonts
font_small = pygame.font.SysFont('arial', 18)
font_large = pygame.font.SysFont('arial', 24, True)

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.direction = RIGHT
        self.head = [WIDTH // 2, HEIGHT // 2]
        self.snake = [self.head[:],
                      [self.head[0] - CELL_SIZE, self.head[1]],
                      [self.head[0] - (2 * CELL_SIZE), self.head[1]]]
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0
        self.game_over = False

    def place_food(self):
        x = random.randint(0, (WIDTH - CELL_SIZE) // CELL_SIZE - 1) * CELL_SIZE
        y = random.randint(0, (HEIGHT - CELL_SIZE) // CELL_SIZE - 1) * CELL_SIZE
        self.food = [x, y]
        if self.food in self.snake:
            self.place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Move
        self.move(action)
        self.snake.insert(0, self.head[:])

        # Check if game over
        reward = 0
        self.game_over = False
        if self.is_collision():
            self.game_over = True
            reward = -10
            return reward, self.game_over, self.score

        # Place new food or move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()

        return reward, self.game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def move(self, action):
        # action: [straight, right, left]
        clock_wise = [RIGHT, DOWN, LEFT, UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # No change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Left turn

        self.direction = new_dir

        x, y = self.head
        if self.direction == RIGHT:
            x = (x + CELL_SIZE) % WIDTH
        elif self.direction == LEFT:
            x = (x - CELL_SIZE) % WIDTH
        elif self.direction == DOWN:
            y = (y + CELL_SIZE) % HEIGHT
        elif self.direction == UP:
            y = (y - CELL_SIZE) % HEIGHT

        self.head = [x, y]

    def clone(self):
        clone_game = SnakeGame()
        clone_game.direction = self.direction
        clone_game.head = self.head[:]
        clone_game.snake = [segment[:] for segment in self.snake]
        clone_game.score = self.score
        clone_game.food = self.food[:]
        clone_game.frame_iteration = self.frame_iteration
        clone_game.game_over = self.game_over
        return clone_game

class Agent:
    def __init__(self):
        self.n_simulations = 100  # Number of simulations per move
        self.max_steps = 50       # Max steps per simulation

    def get_state(self, game):
        head = game.snake[0]
        point_l = [head[0] - CELL_SIZE, head[1]]
        point_r = [head[0] + CELL_SIZE, head[1]]
        point_u = [head[0], head[1] - CELL_SIZE]
        point_d = [head[0], head[1] + CELL_SIZE]

        dir_l = game.direction == LEFT
        dir_r = game.direction == RIGHT
        dir_u = game.direction == UP
        dir_d = game.direction == DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food[0] < game.head[0],  # food left
            game.food[0] > game.head[0],  # food right
            game.food[1] < game.head[1],  # food up
            game.food[1] > game.head[1]   # food down
        ]

        return np.array(state, dtype=int)

    def get_action(self, game):
        possible_actions = [
            [1, 0, 0],  # Straight
            [0, 1, 0],  # Right
            [0, 0, 1]   # Left
        ]
        scores = []
        simulations = []

        # Run simulations for each possible action
        for action in possible_actions:
            total_score = 0
            action_simulations = []
            for _ in range(self.n_simulations):
                simulation_game = game.clone()
                simulation_game.play_step(action)
                simulation_score = self.run_simulation(simulation_game)
                total_score += simulation_score
                action_simulations.append(simulation_game)
            average_score = total_score / self.n_simulations
            scores.append(average_score)
            simulations.append(action_simulations)

        # Choose the action with the highest average score
        best_action_idx = np.argmax(scores)
        best_action = possible_actions[best_action_idx]
        best_simulations = simulations[best_action_idx]
        return best_action, best_simulations, scores

    def run_simulation(self, game):
        steps = 0
        while not game.game_over and steps < self.max_steps:
            action = random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            game.play_step(action)
            steps += 1
        return game.score

def draw_grid():
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (WIDTH, y))

def draw_translucent_snakes(simulations):
    s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for sim_game in simulations:
        for pt in sim_game.snake:
            rect = pygame.Rect(pt[0], pt[1], CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(s, TRANSLUCENT_GREEN, rect)
    screen.blit(s, (0, 0))

def draw_main_snake(snake):
    for idx, pt in enumerate(snake.snake):
        color = GREEN if idx == 0 else LIGHT_GREEN
        pygame.draw.rect(screen, color, pygame.Rect(pt[0], pt[1], CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, DARK_GREEN, pygame.Rect(pt[0], pt[1], CELL_SIZE, CELL_SIZE), 1)

def update_ui(game, simulations, scores):
    screen.fill(BLACK)
    draw_grid()
    draw_translucent_snakes(simulations)
    draw_main_snake(game)
    # Draw food
    pygame.draw.rect(screen, RED, pygame.Rect(game.food[0], game.food[1], CELL_SIZE, CELL_SIZE))
    # Display score and AI info
    text_score = font_large.render(f"Score: {game.score}", True, WHITE)
    screen.blit(text_score, [10, 10])

    # Display AI decision info
    actions = ['Straight', 'Right', 'Left']
    for idx, score in enumerate(scores):
        text = font_small.render(f"Action '{actions[idx]}': Avg Score {score:.2f}", True, WHITE)
        screen.blit(text, [10, 40 + idx * 20])

    pygame.display.flip()

def train():
    agent = Agent()
    game = SnakeGame()
    high_score = 0
    total_games = 0
    total_score = 0

    while True:
        # Get action from agent
        action, simulations, scores = agent.get_action(game)

        # Play one step with the main game
        reward, done, score = game.play_step(action)

        # Update UI
        update_ui(game, simulations, scores)
        clock.tick(30)

        if done:
            total_games += 1
            total_score += score
            if score > high_score:
                high_score = score

            print(f"Game {total_games} | Score: {score} | High Score: {high_score} | Avg Score: {total_score/total_games:.2f}")
            game.reset()

if __name__ == '__main__':
    train()
