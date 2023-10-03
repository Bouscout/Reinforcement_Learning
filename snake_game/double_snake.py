# we will run a snake game with two snake inside collaborating to finish the game

import pygame as pg
import random
from snake_game.utils import tools
from snake_game.snake import Snake

perso = True
action_space = 4
obs_space = 8


WIDTH = 500
HEIGHT = 500

DIMENSIONS = (15, 15)  # size W x H
MAX_BLOCKS_W, MAX_BLOCKS_H = DIMENSIONS

# size of a single block
BLOCK = (WIDTH // DIMENSIONS[0] , HEIGHT // DIMENSIONS[1])
block_w, block_h = BLOCK


def place(block_x, block_y):
    x_coord, y_coord = block_x * block_w, block_y * block_h
    return x_coord, y_coord, *BLOCK

def get_block(x_coord, y_coord):
    block_x, block_y = x_coord // block_w, y_coord // block_h
    return block_x, block_y

GRID = []
for i in range(MAX_BLOCKS_H):
    for j in range(MAX_BLOCKS_W) :
        GRID.append(pg.Rect(place(i, j)))

clock = pg.time.Clock()



print(BLOCK)
helper = tools(MAX_BLOCKS_W, MAX_BLOCKS_H, BLOCK)

class Direction():
    right = (1, 0)
    left = (-1, 0)
    up = (0, -1)
    down = (0, 1)


   

class Snake_Game() :
    def __init__(self) -> None:
        pg.init()
        self.running = True

        self.screen = pg.display.set_mode((WIDTH, HEIGHT)) 

        self.snakes = [Snake(), Snake()]

        # to control the speed of the game
        self.latency_index = 0
        self.latency = 0

        self.food = pg.Rect(place(MAX_BLOCKS_W-1, MAX_BLOCKS_H-1))

        self.input_dict = {
            0 : Direction.right ,
            1 : Direction.left ,
            2 : Direction.up ,
            3 :Direction.down,
        }

        self.draw_vision = False
        self.score = 0

    def drawing(self) :
        self.screen.fill((0, 0, 0))

        def draw_snake(snake):
            snake_body = snake.snake()
            pg.draw.rect(self.screen, "purple", snake.head) # head
            pg.draw.rect(self.screen, "green", snake_body[0]()) # tail
            for block in snake_body[1:] :
                pg.draw.rect(self.screen, 'red', block())

        for snake in self.snakes:
            draw_snake(snake)

        # self.screen.blit()
        pg.draw.rect(self.screen, 'blue', self.food)


        for grid in GRID :
            pg.draw.rect(self.screen, "grey", grid, 1)

        if self.draw_vision :
            for duo in self.snake.vision :
                pg.draw.line(self.screen, "green", *duo, 2)


        pg.display.flip()

    def place_food(self, eaten=False):
        if not eaten :
            return 

        safe_space = False
        while not safe_space :
            food_X = random.randint(0, MAX_BLOCKS_W-1)        
            food_Y = random.randint(0, MAX_BLOCKS_H-1)   

            safe_space = True

            for block in [*self.snakes[0].snake(), self.snakes[0], *self.snakes[1].snake(), self.snakes[1]] :
                if food_X != block.x :
                    continue
                else :
                    if food_Y != block.y :
                        continue
                    else :
                        safe_space = False
                        break
                        
                
        # place the food
        self.food = pg.Rect(place(food_X, food_Y))

    def collision_between_snakes(self):
        def check_collision(snake_1, snake_2):
            head_x, head_y = snake_1.x, snake_1.y
            if head_x == snake_2.x and head_y == snake_2.y :
                return True
            
            for block in snake_2.snake() :
                if head_x == block.x :
                    if head_y == block.y:
                        return True

            return False    
        
        if check_collision(self.snakes[0], self.snakes[1]) :
            return True
        
        if check_collision(self.snakes[1], self.snakes[0]):
            return True
        
        return False
        


    
    def check_consumption(self):
        food_X, food_Y = get_block(self.food.x, self.food.y)
        for snake in self.snakes :
            if snake.x == food_X and snake.y == food_Y :
                return snake
            
        
        return False
    
    def handle_inputs(self, direction=None, human=True):
        if human :
            if pg.key.get_pressed()[pg.K_RIGHT] :
                direction = 5
            elif pg.key.get_pressed()[pg.K_LEFT] :
                direction = 1
            elif pg.key.get_pressed()[pg.K_UP] :
                direction = 2
            elif pg.key.get_pressed()[pg.K_DOWN] :
                direction = 3

            if direction :
                if direction == 5 :
                    direction = 0
        else :
            new_direction = self.input_dict[direction]
            return new_direction

    def game_loop(self, action=None) :
        clock.tick(60)
        for event in pg.event.get():
            if event.type == pg.QUIT :
                self.running =False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_q :
                    self.running = False

                if event.key == pg.K_SPACE :
                    try :
                        new_latency = int(input("enter the latency value, max is 60 : "))
                        self.latency = new_latency

                    except :
                        pass 



        state = None
        reward = None
        collision = None
        made_move = False
        has_won = False
        if self.latency_index >= self.latency :
            reward = 0


            # game logic and player actions goes here
            new_dir_1 = self.handle_inputs(direction=action[0], human=False)
            new_dir_2 = self.handle_inputs(direction=action[1], human=False)

            bad_move_1 = self.snakes[0].move(new_dir=new_dir_1)
            bad_move_2 = self.snakes[1].move(new_dir=new_dir_2)

            snake = self.check_consumption()
            if snake :
                self.score += 1
                has_won = snake.digest_food()
                if not has_won :
                    self.place_food(eaten=True)
                # reward = 15 / (len(self.snake.snake()) + 2)
                reward = 10

                # reward = 15 - (len(self.snake.snake()) + 5)
                # reward = reward if reward >= 0 else 0 

            # checking for collision
            if not has_won :
                if not bad_move_1 or not bad_move_2 :
                    collision = self.snakes[0].check_collisions() or self.snakes[1].check_collisions()
                    if collision or self.collision_between_snakes(): 
                        # print("collision")
                        self.running = False
                        reward = -10 

                else :
                    # print("bad move")
                    reward = -10
                    collision = True

            else :
                self.running = False
                reward = 50
                collision = True
            
            self.drawing()

            state_1 = helper.get_state(self.snakes[0], self.food, self.snakes[1])
            state_2 = helper.get_state(self.snakes[1], self.food, self.snakes[0])
            
            states = [state_1, state_2]
            # state = helper.get_state_image(self.screen)
            # state = helper.state_box(self.snake, self.food)

            made_move = True
            self.latency_index = 0
        else : self.latency_index += 1

        self.drawing()

        # helper.get_state(self.snake, self.food)

        return states, reward, collision, made_move


    def step(self, action):
        made_move = False
        while not made_move :
            state, reward, done, made_move = self.game_loop(action=action)

        return state, reward, done

    def reset(self):
        self.snakes[0].reset()
        self.snakes[1].reset(7, 10)
        self.place_food(eaten=True)
        self.score = 0

        state_1 = helper.get_state(self.snakes[0], self.food, self.snakes[1])
        state_2 = helper.get_state(self.snakes[1], self.food, self.snakes[0])
        # state = helper.get_state_image(self.screen)
        # state = helper.state_box(self.snake, self.food)
        states = [state_1, state_2]
        return states

    def run(self):
        while self.running :
            self.game_loop()

