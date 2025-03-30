import numpy as np
import pickle
import os

class SnakeAI:
    def __init__(self, ga):
        self.ga = ga
        self.population = self.ga.initialize_population()
        self.current_snake_idx = 0
        self.best_score = 0
        self.fitness_scores = [0] * self.ga.population_size
        self.current_weights = None
        self.vision_values = None
        self.last_output = None
        
    def get_move(self, vision_input):
        self.current_weights = self.population[self.current_snake_idx]
        self.vision_values = vision_input
        output = self._forward(vision_input, self.current_weights)
        self.last_output = output
        return np.argmax(output)
    
    def _forward(self, x, weights):
        W1, b1, W2, b2, W3, b3 = weights
        
        # Input to first hidden layer (24 -> 16)
        z1 = np.dot(x, W1) + b1
        a1 = np.maximum(0, z1)  # ReLU activation
        
        # First hidden to second hidden layer (16 -> 16)
        z2 = np.dot(a1, W2) + b2
        a2 = np.maximum(0, z2)  # ReLU activation
        
        # Second hidden to output layer (16 -> 4)
        z3 = np.dot(a2, W3) + b3
        
        # Softmax for output layer
        exp_z3 = np.exp(z3 - np.max(z3))
        output = exp_z3 / np.sum(exp_z3)
        
        return output
    
    def update_fitness(self, score, moves_used):
        # Calculate fitness based on score and survival time
        if score == 0:
            fitness = moves_used
        else:
            fitness = moves_used + (2 ** score) * 1000
            
        self.fitness_scores[self.current_snake_idx] = fitness
        
        if score > self.best_score:
            self.best_score = score
            self.ga.best_score = score
            
        self.current_snake_idx += 1
        
        if self.current_snake_idx >= self.ga.population_size:
            print(f"Generation {self.ga.generation} complete")
            print(f"Best Score: {self.best_score}")
            print(f"Best Fitness: {max(self.fitness_scores)}")
            
            self.population = self.ga.evolve(self.population, self.fitness_scores)
            if self.ga.generation % 5 == 0:
                self.ga.save_population(self.population)
                self.ga.save_best_snake(self.population, self.fitness_scores)
                self.ga.plot_fitness_history()
                
            self.current_snake_idx = 0
            self.fitness_scores = [0] * self.ga.population_size
            
    def save_best_brain(self):
        best_idx = np.argmax(self.fitness_scores)
        best_brain = self.population[best_idx]
        with open('best_brain.pkl', 'wb') as f:
            pickle.dump(best_brain, f)
        print("Best brain saved successfully!")
            
    def load_best_brain(self):
        if os.path.exists('best_brain.pkl'):
            with open('best_brain.pkl', 'rb') as f:
                return pickle.load(f)
        return None