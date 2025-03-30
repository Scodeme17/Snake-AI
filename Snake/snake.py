import numpy as np
from enum import Enum

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Snake:
    def __init__(self, grid_width, grid_height):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.reset()
        
    def reset(self):
        middle_x = self.grid_width // 2
        middle_y = self.grid_height // 2
        self.body = [(middle_x, middle_y), (middle_x, middle_y+1), (middle_x, middle_y+2)]
        self.direction = Direction.UP
        self.food = self._spawn_food()
        self.score = 0
        self.moves_left = 200
        self.moves_without_food = 0
        self.dead = False
        
    def _spawn_food(self):
        while True:
            x = np.random.randint(0, self.grid_width)
            y = np.random.randint(0, self.grid_height)
            if (x, y) not in self.body:
                return (x, y)
    
    def change_direction(self, new_direction):
        if (new_direction == Direction.UP and self.direction != Direction.DOWN) or \
           (new_direction == Direction.DOWN and self.direction != Direction.UP) or \
           (new_direction == Direction.LEFT and self.direction != Direction.RIGHT) or \
           (new_direction == Direction.RIGHT and self.direction != Direction.LEFT):
            self.direction = new_direction
    
    def move(self):
        if self.dead:
            return False
            
        self.moves_left -= 1
        self.moves_without_food += 1
        
        if self.moves_without_food >= 200:
            self.dead = True
            return False
            
        head_x, head_y = self.body[0]
        
        if self.direction == Direction.UP:
            new_head = (head_x, head_y - 1)
        elif self.direction == Direction.DOWN:
            new_head = (head_x, head_y + 1)
        elif self.direction == Direction.LEFT:
            new_head = (head_x - 1, head_y)
        else:
            new_head = (head_x + 1, head_y)
        
        # Check if the snake hit the boundary
        if (new_head[0] < 0 or new_head[0] >= self.grid_width or 
            new_head[1] < 0 or new_head[1] >= self.grid_height):
            self.dead = True
            return False
            
        # Check for self-collision
        if new_head in self.body:
            self.dead = True
            return False
            
        self.body.insert(0, new_head)
        
        # Check if the snake ate the food
        if new_head == self.food:
            self.score += 1
            self.moves_left = min(self.moves_left + 100, 500)
            self.moves_without_food = 0
            self.food = self._spawn_food()
        else:
            self.body.pop()
            
        return True
        
    def get_vision(self):
        head = self.body[0]
        directions = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]
        
        vision = []
        max_dist = max(self.grid_width, self.grid_height)
        
        for dx, dy in directions:
            food_dist = 1.0
            body_dist = 1.0
            wall_dist = 1.0
            
            x, y = head
            step = 0
            
            while True:
                x += dx
                y += dy
                step += 1
                
                # Wall detection
                if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
                    wall_dist = min(1.0, step / max_dist)
                    break
                
                # Body detection
                if (x, y) in self.body and body_dist == 1.0:
                    body_dist = min(1.0, step / max_dist)
                
                # Food detection
                if (x, y) == self.food and food_dist == 1.0:
                    food_dist = min(1.0, step / max_dist)
                
                if step >= max_dist:
                    break
            
            # Add the three vision metrics for this direction
            vision.extend([food_dist, body_dist, wall_dist])
        
        return np.array(vision)