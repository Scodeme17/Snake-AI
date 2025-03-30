import pygame
import numpy as np
from snake import Direction
import imageio
import os

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
LIGHT_GREEN = (50, 255, 50)
DARK_GRAY = (30, 30, 30)
BORDER_COLOR = (100, 100, 100)

class SnakeGameGUI:
    def __init__(self, snake, ai, ga, width=1200, height=800, block_size=20):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake AI with Neural Network")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 18)
        self.big_font = pygame.font.SysFont('Arial', 24)
        
        self.width = width
        self.height = height
        self.block_size = block_size
        
        self.snake = snake
        self.ai = ai
        self.ga = ga
        
        self.running = True
        self.speed = 100
        self.generation_speeds = {0: 60, 10: 200, 50: 600}
        
        self.network_area_rect = pygame.Rect(0, 0, width//2, height)
        self.game_area_rect = pygame.Rect(width//2, 0, width//2, height)
        
        self.button_width = 100
        self.button_height = 40
        self.save_button = pygame.Rect(130, 20, self.button_width, self.button_height)
        self.load_button = pygame.Rect(240, 20, self.button_width, self.button_height)
        self.graph_button = pygame.Rect(350, 20, self.button_width, self.button_height)
        self.save_gif_button = pygame.Rect(460, 20, self.button_width, self.button_height)
        
        self.show_vision = True
        self.show_network = True
        self.gif_frames = []
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_s:
                    self.ai.save_best_brain()
                elif event.key == pygame.K_SPACE:
                    if self.speed == 60:
                        self.speed = 100
                    elif self.speed == 100:
                        self.speed = 200
                    else:
                        self.speed = 300
                elif event.key == pygame.K_v:
                    self.show_vision = not self.show_vision
                elif event.key == pygame.K_n:
                    self.show_network = not self.show_network
                elif event.key == pygame.K_g:
                    self.save_gif()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if self.save_button.collidepoint(mouse_pos):
                    self.ai.save_best_brain()
                elif self.load_button.collidepoint(mouse_pos):
                    best_brain = self.ai.load_best_brain()
                    if best_brain:
                        self.ai.current_weights = best_brain
                        print("Brain loaded!")
                elif self.graph_button.collidepoint(mouse_pos):
                    self.ga.plot_fitness_history()
                elif self.save_gif_button.collidepoint(mouse_pos):
                    self.save_gif()
                    
                plus_button = pygame.Rect(355, 90, 20, 20)
                minus_button = pygame.Rect(380, 90, 20, 20)
                if plus_button.collidepoint(mouse_pos):
                    self.ga.mutation_rate = min(0.5, self.ga.mutation_rate + 0.01)
                    self.ga.default_mutation = self.ga.mutation_rate
                    print(f"Mutation rate increased to {self.ga.mutation_rate:.2f}")
                elif minus_button.collidepoint(mouse_pos):
                    self.ga.mutation_rate = max(0.01, self.ga.mutation_rate - 0.01)
                    self.ga.default_mutation = self.ga.mutation_rate
                    print(f"Mutation rate decreased to {self.ga.mutation_rate:.2f}")
    
    def draw_buttons(self):
        pygame.draw.rect(self.screen, WHITE, self.save_button)
        pygame.draw.rect(self.screen, WHITE, self.load_button)
        pygame.draw.rect(self.screen, WHITE, self.graph_button)
        pygame.draw.rect(self.screen, WHITE, self.save_gif_button)
        
        save_text = self.font.render("Save", True, BLACK)
        load_text = self.font.render("Load", True, BLACK)
        graph_text = self.font.render("Graph", True, BLACK)
        gif_text = self.font.render("Save GIF", True, BLACK)
        
        self.screen.blit(save_text, (self.save_button.x + (self.button_width - save_text.get_width()) // 2, 
                                    self.save_button.y + (self.button_height - save_text.get_height()) // 2))
        self.screen.blit(load_text, (self.load_button.x + (self.button_width - load_text.get_width()) // 2, 
                                    self.load_button.y + (self.button_height - load_text.get_height()) // 2))
        self.screen.blit(graph_text, (self.graph_button.x + (self.button_width - graph_text.get_width()) // 2, 
                                     self.graph_button.y + (self.button_height - graph_text.get_height()) // 2))
        self.screen.blit(gif_text, (self.save_gif_button.x + (self.button_width - gif_text.get_width()) // 2, 
                                   self.save_gif_button.y + (self.button_height - gif_text.get_height()) // 2))
    
    def draw_game(self):
        game_area_x = self.width // 2
        
        # Draw game area background
        pygame.draw.rect(self.screen, BLACK, (game_area_x, 0, self.width//2, self.height))
        
        # Draw game area border
        pygame.draw.rect(self.screen, BORDER_COLOR, (game_area_x, 0, self.width//2, self.height), 3)
        
        # Draw snake
        for i, (x, y) in enumerate(self.snake.body):
            color_val = max(50, 255 - i * 15)
            snake_color = (30, color_val, 30)
            
            rect = pygame.Rect(
                game_area_x + x * self.block_size,
                y * self.block_size,
                self.block_size,
                self.block_size
            )
            pygame.draw.rect(self.screen, snake_color, rect)
            
            pygame.draw.rect(self.screen, (10, 10, 10), rect, 1)
        
        # Draw food
        food_rect = pygame.Rect(
            game_area_x + self.snake.food[0] * self.block_size,
            self.snake.food[1] * self.block_size,
            self.block_size,
            self.block_size
        )
        pygame.draw.rect(self.screen, RED, food_rect)
        
        # Draw score
        score_text = self.big_font.render(f"Score: {self.snake.score}", True, WHITE)
        self.screen.blit(score_text, (game_area_x + 10, 10))
        
        # Draw vision lines if enabled
        if self.show_vision and self.ai.vision_values is not None:
            head_x, head_y = self.snake.body[0]
            head_center = (
                game_area_x + head_x * self.block_size + self.block_size//2,
                head_y * self.block_size + self.block_size//2
            )
            
            directions = [
                (0, -1), (1, -1), (1, 0), (1, 1),
                (0, 1), (-1, 1), (-1, 0), (-1, -1)
            ]
            
            max_dist = max(self.snake.grid_width, self.snake.grid_height)
            
            for i, (dx, dy) in enumerate(directions):
                vision_idx = i * 3  # Updated for 3 values per direction
                food_dist = self.ai.vision_values[vision_idx]
                body_dist = self.ai.vision_values[vision_idx + 1]
                wall_dist = self.ai.vision_values[vision_idx + 2]
                
                distance = max_dist
                if wall_dist < 1.0:
                    distance = min(distance, int(max_dist * wall_dist))
                    
                end_x = head_center[0] + dx * distance * self.block_size
                end_y = head_center[1] + dy * distance * self.block_size
                
                pygame.draw.line(self.screen, (50, 50, 100), head_center, (end_x, end_y), 1)
                
                if food_dist < 1.0:
                    food_x = head_center[0] + dx * int(max_dist * food_dist) * self.block_size
                    food_y = head_center[1] + dy * int(max_dist * food_dist) * self.block_size
                    pygame.draw.circle(self.screen, (255, 0, 0), (int(food_x), int(food_y)), 3)
                
                if body_dist < 1.0:
                    body_x = head_center[0] + dx * int(max_dist * body_dist) * self.block_size
                    body_y = head_center[1] + dy * int(max_dist * body_dist) * self.block_size
                    pygame.draw.circle(self.screen, (0, 255, 0), (int(body_x), int(body_y)), 3)
    
    def draw_network(self):
        # Clear the network area with black background
        pygame.draw.rect(self.screen, BLACK, self.network_area_rect)
        
        # Draw UI buttons
        self.draw_buttons()
        
        # Check if weights are available
        if not self.ai.current_weights or len(self.ai.current_weights) != 6:  # Updated for 3 layers (input, 2 hidden, output)
            text = self.font.render("Neural network visualization not available", True, WHITE)
            self.screen.blit(text, (50, self.height//2))
            return
            
        W1, b1, W2, b2, W3, b3 = self.ai.current_weights
        
        # Draw neural network stats
        gen_text = self.font.render(f"GENERATION: {self.ga.generation}", True, WHITE)
        self.screen.blit(gen_text, (150, 65))
        
        # Draw mutation rate with + and - buttons
        mutation_text = self.font.render(f"MUTATION RATE: {self.ga.mutation_rate:.2f}", True, WHITE)
        self.screen.blit(mutation_text, (150, 90))
        
        # Plus and minus buttons for mutation rate
        plus_button = pygame.Rect(355, 90, 20, 20)
        minus_button = pygame.Rect(380, 90, 20, 20)
        pygame.draw.rect(self.screen, WHITE, plus_button)
        pygame.draw.rect(self.screen, WHITE, minus_button)
        
        plus_text = self.font.render("+", True, BLACK)
        minus_text = self.font.render("-", True, BLACK)
        self.screen.blit(plus_text, (plus_button.x + 6, plus_button.y + 1))
        self.screen.blit(minus_text, (minus_button.x + 7, minus_button.y + 1))
        
        # Current snake info
        score_text = self.font.render(f"CURRENT SCORE: {self.snake.score}", True, WHITE)
        self.screen.blit(score_text, (150, self.height - 90))
        
        highscore_text = self.font.render(f"HIGH SCORE: {self.ai.best_score}", True, WHITE)
        self.screen.blit(highscore_text, (150, self.height - 60))
        
        best_fitness_text = self.font.render(f"BEST FITNESS: {self.ga.best_fitness:.0f}", True, WHITE)
        self.screen.blit(best_fitness_text, (150, self.height - 30))
        
        population_text = self.font.render(f"POPULATION SIZE: {self.ga.population_size}", True, WHITE)
        self.screen.blit(population_text, (150, self.height - 120))
        
        if self.speed == 0:
            speed_text = "SPEED: MAX"
        else:
            speed_text = f"SPEED: {self.speed} FPS"
        speed_display = self.font.render(speed_text, True, WHITE)
        self.screen.blit(speed_display, (150, 115))
        
        # Network visualization parameters
        layers = [24, 16, 8, 4]  # Updated for 24 inputs, 2x16 hidden layers, 4 outputs
        layer_x = [50, 200, 350, 500]  # Adjusted x positions for better spacing
        layer_colors = [WHITE, WHITE, WHITE, WHITE]
        
        # Calculate neuron positions for each layer
        neurons_y = []
        for i, n in enumerate(layers):
            spacing = min(15, (self.height - 200) // (n + 2))
            start_y = (self.height - 120) // 2 - (n * spacing) // 2 + 120
            neurons_y.append([start_y + j * spacing for j in range(n)])
        
        # Map output indices to direction names for better visualization
        direction_names = ["UP", "RIGHT", "DOWN", "LEFT"]
        
        # Draw connections if network visualization is enabled
        if self.show_network and self.ai.last_output is not None:
            max_output_idx = np.argmax(self.ai.last_output)
            
            # Draw connections from layer 2 to output layer
            for i in range(layers[2]):
                for j in range(layers[3]):
                    weight = W3[i, j]
                    # Highlight the connection to the chosen output
                    if j == max_output_idx:
                        color = LIGHT_GREEN if weight > 0 else RED
                        width = max(1, min(4, int(abs(weight) * 5)))
                    else:
                        if abs(weight) < 0.1:
                            continue
                        color = BLUE if weight > 0 else RED
                        width = max(1, min(2, int(abs(weight) * 3)))
                    
                    pygame.draw.line(
                        self.screen,
                        color,
                        (layer_x[2], neurons_y[2][i]),
                        (layer_x[3], neurons_y[3][j]),
                        width
                    )
            
            # Draw connections from layer 1 to layer 2 (sample to avoid clutter)
            connection_sample_rate = 0.1
            for i in range(layers[1]):
                for j in range(layers[2]):
                    if np.random.rand() > connection_sample_rate:
                        continue
                        
                    weight = W2[i, j]
                    if abs(weight) < 0.2:
                        continue
                        
                    color = BLUE if weight > 0 else RED
                    width = max(1, min(2, int(abs(weight) * 2)))
                    
                    pygame.draw.line(
                        self.screen,
                        color,
                        (layer_x[1], neurons_y[1][i]),
                        (layer_x[2], neurons_y[2][j]),
                        width
                    )
            
            # Draw connections from input layer to layer 1 (sample to avoid clutter)
            connection_sample_rate = 0.05
            for i in range(layers[0]):
                for j in range(layers[1]):
                    if np.random.rand() > connection_sample_rate:
                        continue
                        
                    weight = W1[i, j]
                    if abs(weight) < 0.2:
                        continue
                        
                    color = BLUE if weight > 0 else RED
                    width = max(1, min(2, int(abs(weight) * 2)))
                    
                    pygame.draw.line(
                        self.screen,
                        color,
                        (layer_x[0], neurons_y[0][i]),
                        (layer_x[1], neurons_y[1][j]),
                        width
                    )
        
        # Determine active input neurons based on vision values
        active_inputs = []
        if self.ai.vision_values is not None:
            for i in range(len(self.ai.vision_values)):
                # For each of the 8 directions, we have 3 values
                # Values represent distances (lower is closer)
                active_inputs.append(self.ai.vision_values[i] < 0.5)
        
        # Draw all neurons
        for i, layer in enumerate(layers):
            for j in range(layer):
                radius = 4 if i == 0 else 6
                color = layer_colors[i]
                
                # Highlight input layer nodes based on vision
                if i == 0 and j < len(active_inputs):
                    if active_inputs[j]:
                        color = LIGHT_GREEN
                        radius = 5
                
                # Highlight output layer based on decision
                if i == 3 and self.ai.last_output is not None:
                    if j == np.argmax(self.ai.last_output):
                        color = LIGHT_GREEN
                        radius = 8
                
                pygame.draw.circle(self.screen, color, (layer_x[i], neurons_y[i][j]), radius)
              
    def save_gif(self, filename="snake_game.gif", duration=0.1):
        if not self.gif_frames:
            print("No frames to save.")
            return
            
        imageio.mimsave(filename, self.gif_frames, duration=duration)
        print(f"GIF saved to {filename}")
        self.gif_frames = []
    
    def update(self):
        # Automatically adjust speed based on generation
        for gen, speed in self.generation_speeds.items():
            if self.ga.generation == gen and self.speed != 0:
                self.speed = speed
                print(f"Speed automatically adjusted to {speed} FPS at generation {gen}")
        
        # Get vision data and determine move
        vision = self.snake.get_vision()
        move = self.ai.get_move(vision)
        
        # Apply the move and update the snake
        self.snake.change_direction(Direction(move))
        alive = self.snake.move()
        
        # If snake died, update fitness and reset
        if not alive:
            self.ai.update_fitness(self.snake.score, self.snake.moves_left)
            self.snake.reset()
        
        # Draw everything
        self.screen.fill(BLACK)
        if self.show_network:
            self.draw_network()
        self.draw_game()
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        if self.speed > 0:
            self.clock.tick(self.speed)
            
        # Capture frame for GIF if both visualizations are enabled
        if self.show_vision and self.show_network and len(self.gif_frames) < 200:  # Limit frames to avoid memory issues
            pygame.image.save(self.screen, "temp.png")
            self.gif_frames.append(imageio.imread("temp.png"))
            if os.path.exists("temp.png"):  # Check if file exists before removing
                try:
                    os.remove("temp.png")
                except:
                    pass