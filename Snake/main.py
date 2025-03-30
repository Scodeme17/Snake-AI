from genetic import GeneticAlgorithm
from neural_network import SnakeAI
from snake import Snake
from gui import SnakeGameGUI
import matplotlib
import sys
import pygame

# Constants
WIDTH = 1200
HEIGHT = 800
GRID_WIDTH = 40
GRID_HEIGHT = 30
BLOCK_SIZE = 20
FPS = 100

def main():
    # Updated population size to 2000 as per description
    ga = GeneticAlgorithm(population_size=3000, mutation_rate=0.02)
    snake = Snake(GRID_WIDTH, GRID_HEIGHT)
    ai = SnakeAI(ga)
    
    gui = SnakeGameGUI(snake, ai, ga, width=WIDTH, height=HEIGHT, block_size=BLOCK_SIZE)
    
    while gui.running:
        gui.handle_events()
        gui.update()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()