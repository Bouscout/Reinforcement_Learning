import pygame as pg
import random
from snake_game.agent_tools import tools
from snake_game.snake import Snake

perso = True
action_space = 4
obs_space = 8


WIDTH = 500
HEIGHT = 500

DIMENSIONS = (6, 6)  # size W x H
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

        self.snake = Snake()

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

        snake = self.snake.snake()
        pg.draw.rect(self.screen, "purple", self.snake.head)
        pg.draw.rect(self.screen, "green", snake[0]())
        for block in snake[1:] :
            pg.draw.rect(self.screen, 'red', block())
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

            for block in [*self.snake.snake(), self.snake] :
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
        


    
    def check_consumption(self):
        food_X, food_Y = get_block(self.food.x, self.food.y)
        if self.snake.x == food_X and self.snake.y == food_Y :
            return True
        
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
            if len(self.snake.snake()) < ((MAX_BLOCKS_H * MAX_BLOCKS_W) // 2) - 1 :
                reward = -0.2


            # game logic and player actions goes here
            new_dir = self.handle_inputs(direction=action, human=False)

            bad_move = self.snake.move(new_dir=new_dir)


            if self.check_consumption() :
                self.score += 1
                has_won = self.snake.digest_food()
                if not has_won :
                    self.place_food(eaten=True)
                
                if len(self.snake.snake()) > ((MAX_BLOCKS_H * MAX_BLOCKS_W) // 2) - 1  or len(self.snake.snake()) > 15:
                    if random.randint(0, 15) <= 10 :
                        reward = 0
                    else : reward = 10
                else :
                    reward = 10
                
            # checking for collision
            if not has_won :
                if not bad_move :
                    collision = self.snake.check_collisions()
                    if collision : 
                        # print("collision")
                        self.running = False
                        reward = -10 

                else :
                    # print("bad move")
                    reward = -10
                    collision = True

            else :
                self.running = False
                reward = 20
                collision = True
            
            self.drawing()

            state = helper.get_state(self.snake, self.food)
            # state = helper.get_state_image(self.screen)
            # state = helper.state_box(self.snake, self.food)

            made_move = True
            self.latency_index = 0
        else : self.latency_index += 1

        self.drawing()

        # helper.get_state(self.snake, self.food)

        return state, reward, collision, made_move


    def step(self, action):
        made_move = False
        while not made_move :
            state, reward, done, made_move = self.game_loop(action=action)

        return state, reward, done

    def reset(self):
        self.snake.reset()
        self.place_food(eaten=True)
        self.score = 0

        state = helper.get_state(self.snake, self.food)
        # state = helper.get_state_image(self.screen)
        # state = helper.state_box(self.snake, self.food)

        return state

    def run(self):
        while self.running :
            self.game_loop()

