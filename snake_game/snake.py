# this file handle all the logics for the snake and its component blocks
from typing import Any
import pygame as pg

# hyperparameters

WIDTH = 500
HEIGHT = 500

DIMENSIONS = (6, 6)  # size W x H
MAX_BLOCKS_W, MAX_BLOCKS_H = DIMENSIONS


def place(block_x, block_y):
    x_coord, y_coord = block_x * block_w, block_y * block_h
    return x_coord, y_coord, *BLOCK

def get_block(x_coord, y_coord):
    block_x, block_y = x_coord // block_w, y_coord // block_h
    return block_x, block_y

# size of a single block
BLOCK = (WIDTH // DIMENSIONS[0] , HEIGHT // DIMENSIONS[1])
block_w, block_h = BLOCK

class Direction():
    right = (1, 0)
    left = (-1, 0)
    up = (0, -1)
    down = (0, 1)

# class to handle the building blocks of the snake
class Block():
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __call__(self, x=None, y=None):
       if x :
           return pg.Rect(place(x, y))
       
       return pg.Rect(place(self.x, self.y)) 
    
    def __repr__(self) -> str:
        return f"Block coord({self.x}, {self.y})"

    def coordinate(self):
        x_coord, y_coord, *extra = place(self.x, self.y)
        return x_coord, y_coord
    

    

class Snake():
    def __init__(self) -> None:
       
       self.reset()

    def __repr__(self) -> str:
        return f"snake coord({self.x}, {self.y})"

    def reset(self, x=None, y=None):
        self.direction = Direction.right
        if x :
            self.x = x
            self.y = y
        else :
            self.x = MAX_BLOCKS_W // 2
            # self.x = 1
            self.y = MAX_BLOCKS_H // 2

        # initializing the snake array
        self.index = -1
        self.blocks = [0 for _ in range(MAX_BLOCKS_W * MAX_BLOCKS_H)] # max size of the snake array
        
        self.head = pg.Rect(place(self.x, self.y))
        self.tail = (self.x-1, self.y) # last tail position
        self.blocks[self.index] = Block(self.x-1, self.y) 


        self.vision = []

    def move(self, new_dir=None):
        if new_dir :
            new_x, new_y = new_dir
            old_x, old_y = self.direction

            if (new_x + old_x) == 0 and (new_y + old_y) == 0 :
                return True
            
            self.direction = new_dir


        size = len(self.blocks)
        snake_indices = (size + self.index, size) 

         # tail last position
        self.tail = (self.blocks[self.index].x, self.blocks[self.index].y)

        for i in range(size+self.index, size - 1) :
            next_block = self.blocks[i+1]

            self.blocks[i].x = next_block.x 
            self.blocks[i].y = next_block.y

        self.blocks[-1].x = self.x
        self.blocks[-1].y = self.y

        # moving the head
        move_x, move_y = self.direction
        self.x += move_x
        self.y += move_y

        
        # self.blocks[-1].x, self.blocks[-1].y, *extra = place(self.x, self.y)

    def check_collisions(self):
        # out of bound collision
        if self.x >= MAX_BLOCKS_W or self.x < 0:
            return True
        
        if self.y >= MAX_BLOCKS_H or self.y < 0 :
            return True
        
        snake = self.blocks[self.index:]
        # head_x, head_y, *extra = place(self.x, self.y)

        for block in snake :
            if self.x == block.x and self.y == block.y :
                return True
            
        return False

    def digest_food(self):
        tail_x, tail_y = self.tail

        self.index -= 1
        # self.blocks[self.index] = pg.Rect(last_block.x, last_block.y, *BLOCK)
        self.blocks[self.index] = Block(tail_x, tail_y)

        # game completed
        if len(self.snake()) >= (MAX_BLOCKS_W * MAX_BLOCKS_H) - 1:
             return True
        
        else : 
            return False
        

    def snake(self):
        self.head.x, self.head.y, *extra = place(self.x, self.y)
        return self.blocks[self.index:]
    
   
