# 🐍 Snake AI with Neural Networks 🧠

A Python implementation of the classic Snake game that trains a neural network using genetic algorithms to play autonomously.

![Snake AI in Action](screenshot.png)

## 🌟 Features

- 🎮 Classic Snake game implementation
- 🧬 Genetic algorithm for neural network training
- 🔄 Multiple generations of evolution
- 📊 Real-time neural network visualization
- 👁️ Vision system for the snake to perceive its environment
- 📈 Fitness tracking and visualization across generations
- 💾 Save/Load functionality for trained models
- 🎥 GIF recording capability to showcase AI behavior

## 🛠️ Technology Stack

- Python 3.x
- Pygame for graphics and user interface
- NumPy for neural network and mathematical operations
- Matplotlib for fitness history visualization
- Pickle for model persistence
- Imageio for GIF generation

## 🧠 Neural Network Architecture

The AI uses a multi-layer neural network:
- Input layer: 24 neurons (8 directions × 3 vision features per direction)
- Hidden layer 1: 16 neurons 
- Hidden layer 2: 8 neurons
- Output layer: 4 neurons (representing UP, RIGHT, DOWN, LEFT)

## 🧬 Genetic Algorithm

The genetic algorithm includes:
- Tournament selection
- Crossover for breeding new snakes
- Mutation to introduce genetic diversity
- Elitism to preserve the best performers
- Adaptive mutation rate to prevent stagnation

## 🚀 Getting Started

### Prerequisites

```bash
pip install pygame numpy matplotlib imageio
```

### Running the Game

```bash
python main.py
```

## 🎮 Controls

- **Space**: Change simulation speed
- **S**: Save best brain
- **V**: Toggle vision lines
- **N**: Toggle neural network visualization
- **G**: Save current gameplay as GIF
- **Esc**: Quit the game

## 📋 GUI Elements

- **Save Button**: Save the best performing neural network
- **Load Button**: Load a previously saved neural network
- **Graph Button**: Generate and save a graph of fitness history
- **Save GIF Button**: Save a GIF of the current gameplay
- **+/- Buttons**: Adjust mutation rate

## 📊 Stats Displayed

- Generation count
- Current mutation rate
- Current score
- High score
- Best fitness achieved
- Population size
- Simulation speed

## 📂 Project Structure

- `main.py` - Entry point
- `snake.py` - Snake game logic and mechanics
- `neural_network.py` - AI implementation
- `genetic.py` - Genetic algorithm implementation
- `gui.py` - User interface and visualization

## 🧪 How It Works

1. The snake uses a vision system that looks in 8 directions
2. For each direction, it detects distance to food, body parts, and walls
3. These 24 inputs feed into the neural network
4. The network produces 4 outputs representing possible movement directions
5. The highest output value determines the snake's next move
6. Fitness is calculated based on survival time and food eaten
7. The best performing snakes breed to create the next generation

## 📝 Training Tips

- Larger population sizes generally lead to faster learning but require more computational resources
- Higher mutation rates promote exploration (good for early training)
- Lower mutation rates promote exploitation (good for refining behavior)
- The adaptive mutation mechanism helps overcome local maxima

## 📈 Performance

After approximately 50 generations with a population size of 3000, the snake typically learns to:
- Effectively seek food
- Avoid walls
- Avoid its own body
- Plan simple paths

## 🤝 Contributing

Contributions are welcome! Some ideas for improvements:
- Add different training environments
- Implement more advanced neural network architectures
- Create challenges with obstacles
- Optimize performance for larger populations
- Add multiplayer competition between trained models

## 🙏 Acknowledgments

- Inspired by various AI and machine learning projects in game environments
- Built upon the classic Snake game concept
