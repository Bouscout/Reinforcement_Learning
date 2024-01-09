import numpy as np
import math
import cv2
import pygame as pg

class tools:
    def __init__(self, w, h, BLOCK) -> None:
        self.max_w = w
        self.max_h = h

        self.BLOCK = BLOCK
        self.block_w, self.block_h = BLOCK

    def get_state_image(self,screen,  input_width=50, input_height=50):
        game_surface = pg.surfarray.array3d(screen)
        game_surface = cv2.resize(game_surface, (input_width, input_height))

        gray_image = cv2.cvtColor(game_surface, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("snake.png", gray_image)

        game_surface = gray_image / 255.0  # Normalize to [0, 1]

        # Convert to NumPy array
        game_array = np.array(game_surface, dtype=np.float32)


        return game_array[None, :]

    # convert block coordinate to coordinate
    def place(self, block_x, block_y):
        x_coord, y_coord = block_x * self.block_w, block_y * self.block_h
        return x_coord+(self.block_w/2), y_coord+(self.block_h/2)
        # return x_coord, y_coord
    
    # convert coordinate to block coordinate
    def get_block(self, x_coord, y_coord):
        block_x, block_y = x_coord // self.block_w, y_coord // self.block_h
        return block_x, block_y

    # this function will receive a distance variable in blocks unit
    # it will convert it to a distance variable between [0, 1] with one being end to end of the board
    def get_distance(self, *args, x_axis=True) -> list:
        distances = []
        for elem in args :

            # checking if it's with respect to the x axis
            # if max_w == max_h the variable x_axis don't matter
            if x_axis :
                distances.append(elem / self.max_w)
            else :
                distances.append(elem / self.max_h)

        return distances
    



    def get_state(self, snake, food) :
        possibilities = 9
        bound_x = self.max_w 
        bound_y= self.max_h 

        snake_blocks = [(block.x, block.y) for block in snake.snake()]

        snake_length = len(snake.snake()) / ((self.max_w * self.max_h))

        # regular_danger_distance, diag_danger_distance = self.get_danger_distances(snake, food)

        state = self.get_danger_distances(snake, food)

        return state

        dist_right, dist_left, dist_up, dist_down = regular_danger_distance
        top_right, top_left, bottom_right, bottom_left = diag_danger_distance
       
        right = dist_right / bound_x
        left = dist_left / bound_x
        up = dist_up / bound_y
        down = dist_down / bound_y

        def check_proximity(primary, secondary, offset):
            for pt_x, pt_y in snake_blocks :
                if primary + offset == pt_x and secondary == pt_y :
                    return True

            return False

        head_x = snake.x
        head_y = snake.y

        food_x, food_y = self.get_block(food.x, food.y) 

        food_dist_x = food_x - head_x
        food_dist_y = food_y - head_y

        base = (self.place(head_x, head_y))

        snake.vision = [
            (base, (self.place(head_x + dist_right+1, head_y))), # right
            (base, (self.place(head_x - dist_left-1, head_y))), # left
            (base, (self.place(head_x, head_y - dist_up-1))), # top
            (base, (self.place(head_x, head_y + dist_down+1))), # down

            (base, (self.place(head_x + top_right, head_y - top_right))), # top right
            (base, (self.place(head_x - top_left, head_y - top_left))), # top left
            (base, (self.place(head_x + bottom_right, head_y + bottom_right))), # bottom right
            (base, (self.place(head_x - bottom_left, head_y + bottom_left))), # bottom left
        ] 


        state = np.array([
        # find the location of the food
        # food_x > head_x , # right 
        # food_x < head_x , # left
        # food_y < head_y , # up
        # food_y > head_y , # down
        food_dist_x / bound_x,
        food_dist_y / bound_y,

        # find the danger
        # on right
        # head_x +1 == bound_x or 
        # check_proximity(primary=head_x, secondary=head_y ,offset=1) ,

        # # on left
        # head_x - 1 == -1 or 
        # check_proximity(primary=head_x, secondary=head_y, offset=-1),

        # #up
        # head_y - 1 == -1 or
        # check_proximity(primary=head_x, secondary=head_y-1 ,offset=0), 

        # # down
        # head_y + 1 == bound_y or
        # check_proximity(primary=head_x, secondary=head_y+1, offset=0),
        
        # distances
        right,
        left,
        up, 
        down,

        # diag distances
        top_right / bound_x,
        top_left / bound_x,
        bottom_right / bound_x,
        bottom_left / bound_x,


        ])

        # making sure that the back direction is considered danger
        # dir_x, dir_y = snake.direction
        # if dir_x == 1 :
        #     state[5] = 1
        # elif dir_x == -1 :
        #     state[4] = 1 

        # if dir_y == 1 :
        #     state[6] = 1
        # elif dir_y == -1  :
        #     state[7] = 1

        state = state.astype(np.float32)

        self.get_danger_distances(snake, food)

        return state
        
    def get_danger_distances(self, snake, food) :

        bound_x = self.max_w - 1
        bound_y = self.max_h - 1
        # create a dict with all the occupied distances
        danger_right = bound_x - snake.x
        danger_left = snake.x
        danger_up = snake.y
        danger_down = bound_y - snake.y

        danger_top_right = min(danger_right, danger_up)
        danger_top_left = min(danger_left, danger_up)
        danger_bottom_right = min(danger_right, danger_down)
        danger_bottom_left = min(danger_left, danger_down)

        head_x , head_y= snake.x, snake.y
        dict_x = {}
        dict_y = {}
        for block in snake.snake() :
            if block.x not in dict_x :
                dict_x.setdefault(block.x, [block])
            else :
                dict_x[block.x].append(block)

            if block.y not in dict_y:
                dict_y.setdefault(block.y, [block])
            else :
                dict_y[block.y].append(block)

        # find relative to the head :
        # distance to the right danger
        def same_level(reference, blocks, x_axis=True):
            for block in blocks :
                if x_axis :
                    if block.x == reference :
                        return True
                
                else :
                    if block.y == reference :
                        return True

            return False

        
        def evaluate_danger(start, end, step, axis_dict, other_axis, x_axis=True, index=0):
            distance = 0
            target = np.zeros((2)) # either a snake, or food

            for i in range(start, end+step, step) :
                # check if it is a the snake
                if i in axis_dict :
                    if same_level(other_axis, axis_dict[i] ,x_axis=x_axis):
                        target[0] = 1
                        state_target[index] = target
                        return distance
                    
                # check it is food
                # if x_axis : check_2, check_1 = self.get_block(food.x, food.y)
                # else : check_1, check_2 = self.get_block(food.x, food.y)

                # if i == check_1 :
                #     if other_axis == check_2 :
                #         target[1] = 1
                #         state_target[index] = target
                #         return distance


                distance += 1

            # in case no triger, it measns it is a wall
            target[1] = 1
            state_target[index] = target
            
            return None

        def evaluate_diag_danger(start, end, step, axis_dict, diag_increment, other_axis, x_axis=True, index=0):
            distance = 0
            up = 0
            target = np.zeros((2))
            for i in range(start, end+step, step):
                up += diag_increment
                # check if it is a snake
                if i in axis_dict :
                    if same_level(other_axis + up, axis_dict[i], x_axis=x_axis) :
                        target[0] = 1
                        state_target[index] = target
                        return distance
                
                # check if it is food
                # if x_axis : check_2, check_1 = self.get_block(food.x, food.y)
                # else : check_1, check_2 = self.get_block(food.x, food.y)

                # if i == check_1 :
                #     if other_axis + up == check_2 :
                #         target[1] = 1
                #         state_target[index] = target
                #         return distance
                

                distance += 1

            # in case none of the previous trigered, means it is a wall
            target[1] = 1
            state_target[index] = target

            return None
                    
        state_target = np.zeros((8, 2))
        

        # danger right
        potential_distance = evaluate_danger(snake.x+1, bound_x, 1, dict_x, head_y, x_axis=False, index=0)
        danger_right = potential_distance if isinstance(potential_distance, int) else danger_right

        # distance to left danger
        potential_distance = evaluate_danger(snake.x - 1, 0, -1, dict_x, head_y, x_axis=False, index=1)
        danger_left = potential_distance if isinstance(potential_distance, int) else danger_left

        # distance to up danger
        potential_distance = evaluate_danger(snake.y-1, 0, -1, dict_y, head_x, x_axis=True, index=2)
        danger_up = potential_distance if isinstance(potential_distance, int) else danger_up

        # distance to down danger
        potential_distance = evaluate_danger(snake.y+1, bound_y, 1, dict_y, head_x, x_axis=True, index=3)
        danger_down = potential_distance if isinstance(potential_distance, int) else danger_down

       
        
        # diag top right
        potential_distance = evaluate_diag_danger(head_x+1, bound_x, 1, dict_x, -1, head_y, x_axis=False, index=4)
        danger_top_right = potential_distance if isinstance(potential_distance, int) else danger_top_right

        # diag top left
        potential_distance = evaluate_diag_danger(head_x-1, 0, -1, dict_x, -1, head_y, x_axis=False, index=5)
        danger_top_left = potential_distance if isinstance(potential_distance, int) else danger_top_left

        # diag bottom right
        potential_distance = evaluate_diag_danger(head_x+1, bound_x, 1, dict_x, 1, head_y, x_axis=False, index=6)
        danger_bottom_right = potential_distance if isinstance(potential_distance, int) else danger_bottom_right

        # diag bottom left 
        potential_distance = evaluate_diag_danger(head_x-1, 0, -1, dict_x, 1, head_y, x_axis=False, index=7)
        danger_bottom_left = potential_distance if isinstance(potential_distance, int) else danger_bottom_left

        diag_collections = (danger_top_right, danger_top_left, danger_bottom_right, danger_bottom_left)
        direct_danger = (danger_right, danger_left, danger_up, danger_down)
       
       
        head_direction = [
            snake.direction == (1, 0), # right
            snake.direction == (-1, 0), # left
            snake.direction == (0, -1), # up
            snake.direction == (0, 1), # down
        ]
        snake_array = snake.snake()
        tail = snake_array[0]
        if len(snake_array) > 1:
            nx_block_X, nx_block_Y = snake_array[1].x, snake_array[1].y
        else :
            nx_block_X, nx_block_Y = head_x, head_y

        tail_move = (nx_block_X - tail.x, nx_block_Y - tail.y)

        tail_direction = [
            tail_move == (1, 0), # right
            tail_move == (-1, 0), # left
            tail_move == (0, -1), # up
            tail_move == (0, 1), # down
        ]

        # tail_position = [
        #     tail.x >= head_x, # right
        #     tail.x < head_x, #left
        #     tail.y < head_y , # up
        #     tail.y >= head_y , # down
        # ]

        food_x, food_y = self.get_block(food.x, food.y)

        food_dist_x = (head_x - food_x) / self.max_w
        food_dist_y = (head_y - food_y) / self.max_h
       
        super_state = np.zeros((35))
        super_state[:4] = np.array(direct_danger) / self.max_w
        super_state[4:8] = np.array(diag_collections) / self.max_w
        super_state[8] = food_dist_x
        super_state[9] = food_dist_y

        super_state[10:26] = np.array(state_target).reshape(-1)
        

        super_state[26:30] = np.array(head_direction)
        super_state[30:34] = np.array(tail_direction)

        snake_length = len(snake.snake()) / ((self.max_w * self.max_w) - 1)
        super_state[34] = snake_length

        return super_state


    
    def state_box(self, snake, food):
        # state = [0 for _ in range(self.max_w * self.max_h)]
        table = self.max_w * self.max_h
        state = [[0 for _ in range(self.max_w)] for _ in range(self.max_h)]
        # state_numpy = np.zeros(((self.max_w * self.max_h) + 4))

        for block in snake.snake() :
            state[block.y][block.x] = 1

        if (snake.x >= 0 and snake.x < self.max_w) and (snake.y >= 0 and snake.y < self.max_h) : 
            state[snake.y][snake.x] = 0.5

        food_x, food_y = self.get_block(food.x, food.y)
        state[food_y][food_x] = -1

        # state_numpy = np.array(state)[None, :, :]
        state_numpy = np.array(state).reshape(-1)

      
        return state_numpy