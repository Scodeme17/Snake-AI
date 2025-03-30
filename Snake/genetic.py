import numpy as np
import matplotlib
import os
import pickle

matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, population_size=3000, mutation_rate=0.02, crossover_rate=0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation = 0
        self.fitness_history = []
        self.best_fitness = 0
        self.best_score = 0
        self.same_best_count = 0
        self.default_mutation = mutation_rate
        self.best_fitness_per_gen = []

    def initialize_population(self):
        population = []
        saved_pop = self.load_population()
        if saved_pop:
            print("Loaded saved population.")
            return saved_pop
            
        print("Creating new population...")
        for _ in range(self.population_size):
            # Input layer (24) to first hidden layer (16)
            W1 = np.random.randn(24, 16) * np.sqrt(2 / 40) 
            b1 = np.zeros(16)

            # First hidden layer (16) to second hidden layer (16)
            W2 = np.random.randn(16, 8) * np.sqrt(2 / (16 + 8))
            b2 = np.zeros(8)

            # Second hidden layer (16) to output layer (4)
            W3 = np.random.randn(8, 4) * np.sqrt(2 / 12)
            b3 = np.zeros(4)

            population.append((W1, b1, W2, b2, W3, b3))

        return population
    
    def evolve(self, population, fitness_scores):
        new_population = []
        
        # Elitism - keep the best individuals
        elite_count = max(1, int(0.05 * self.population_size))
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        elites = [population[idx] for idx in elite_indices]
        new_population.extend(elites)
        
        current_best = max(fitness_scores)
        self.best_fitness_per_gen.append(current_best)
        
        # Check for stagnation and adjust mutation rate if needed
        if current_best <= self.best_fitness:
            self.same_best_count += 1
            if self.same_best_count > 3:
                self.mutation_rate = min(0.3, self.mutation_rate * 1.2)
                print(f"Stagnation detected! Increased mutation rate to {self.mutation_rate}")
        else:
            self.same_best_count = 0
            self.mutation_rate = self.default_mutation
            self.best_fitness = current_best
        
        # Create new population through selection, crossover, and mutation
        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection(population, fitness_scores, k=5)
            parent2 = self._tournament_selection(population, fitness_scores, k=5)
            
            if np.random.rand() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1
                
            child = self._mutate(child)
            new_population.append(child)
        
        self.generation += 1
        avg_fitness = np.mean(fitness_scores)
        self.fitness_history.append((current_best, avg_fitness))
        
        return new_population

    def _tournament_selection(self, population, fitness_scores, k=5):
        indices = np.random.choice(len(population), k)
        best_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
        return population[best_idx]

    def _crossover(self, parent1, parent2):
        child = []
        for i, (p1, p2) in enumerate(zip(parent1, parent2)):
            if len(p1.shape) == 2:
                rows, cols = p1.shape
                crossover_row = np.random.randint(0, rows)
                
                child_param = p1.copy()
                child_param[crossover_row:] = p2[crossover_row:]
            else:
                weight = np.random.rand()
                child_param = weight * p1 + (1 - weight) * p2
            
            child.append(child_param)
        return tuple(child)

    def _mutate(self, individual):
        mutated = []
        for param in individual:
            mask = np.random.rand(*param.shape) < self.mutation_rate
            mutation = np.random.normal(0, 0.2, param.shape)
            mutated_param = param + mutation * mask
            mutated.append(mutated_param)
        return tuple(mutated)

    def save_population(self, population, filename="population.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(population, f)
        print(f"Population saved to {filename}")

    def save_best_snake(self, population, fitness_scores):
        best_idx = np.argmax(fitness_scores)
        best_brain = population[best_idx]
        
        if fitness_scores[best_idx] >= self.best_fitness:
            with open('best_brain_auto.pkl', 'wb') as f:
                pickle.dump(best_brain, f)
            print(f"New best snake saved automatically! Fitness: {fitness_scores[best_idx]}")

    def load_population(self, filename="population.pkl"):
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading population: {e}")
                return None
        return None

    def plot_fitness_history(self):
        if len(self.fitness_history) > 0:
            best_fitness, avg_fitness = zip(*self.fitness_history)
            plt.figure(figsize=(10, 6))
            plt.plot(best_fitness, label='Best Fitness')
            plt.plot(avg_fitness, label='Average Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Fitness over Generations')
            plt.legend()
            plt.grid(True)
            plt.savefig('fitness_history.png')
            plt.close()
            print("Fitness history plot generated!")