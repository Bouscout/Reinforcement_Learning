import pygame as pg
import numpy as np

class tools:
    def __init__(self, w, h, BLOCK) -> None:
        self.max_w = w
        self.max_h = h

        self.BLOCK = BLOCK
        self.block_w, self.block_h = BLOCK


    # convert block coordinate to coordinate
    def place(self, block_x, block_y):
        x_coord, y_coord = block_x * self.block_w, block_y * self.block_h
        return x_coord+(self.block_w/2), y_coord+(self.block_h/2)
        # return x_coord, y_coord
    
    # convert coordinate to block coordinate
    def get_block(self, x_coord, y_coord):
        block_x, block_y = x_coord // self.block_w, y_coord // self.block_h
        return block_x, block_y
    
    def get_state(self, snake, food, other_snake):
        bound_x = self.max_w - 1
        bound_y = self.max_h - 1

        # danger distances relative to the wall
        danger_right = bound_x - snake.x
        danger_left = snake.x
        danger_up = snake.y
        danger_down = bound_y - snake.y

        danger_top_right = min(danger_right, danger_up)
        danger_top_left = min(danger_left, danger_up)
        danger_bottom_right = min(danger_right, danger_down)
        danger_bottom_left = min(danger_left, danger_down)

        head_x , head_y= snake.x, snake.y

        # create a dict with all the occupied distances
        dict_x = {}
        dict_y = {}
        def populate_dict(chosen_snake):
            for block in chosen_snake.snake() :
                if block.x not in dict_x :
                    dict_x.setdefault(block.x, [block])
                else :
                    dict_x[block.x].append(block)

                if block.y not in dict_y:
                    dict_y.setdefault(block.y, [block])
                else :
                    dict_y[block.y].append(block)

        populate_dict(snake)
        populate_dict(other_snake)

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

        other_snake_head_direction = [
            other_snake.direction == (1, 0), # right
            other_snake.direction == (-1, 0), # left
            other_snake.direction == (0, -1), # up
            other_snake.direction == (0, 1), # down
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

        snake_array = other_snake.snake()
        if len(snake_array) > 1:
            nx_block_X, nx_block_Y = snake_array[1].x, snake_array[1].y
        else :
            nx_block_X, nx_block_Y = head_x, head_y
        other_tail_move = (nx_block_X - tail.x, nx_block_Y - tail.y)

        other_tail_direction = [
            other_tail_move == (1, 0), # right
            other_tail_move == (-1, 0), # left
            other_tail_move == (0, -1), # up
            other_tail_move == (0, 1), # down
        ]


        food_x, food_y = self.get_block(food.x, food.y)

        food_dist_x = (head_x - food_x) / self.max_w
        food_dist_y = (head_y - food_y) / self.max_h
       
        super_state = np.zeros((42))
        super_state[:4] = np.array(direct_danger) / self.max_w
        super_state[4:8] = np.array(diag_collections) / self.max_w
        super_state[8] = food_dist_x
        super_state[9] = food_dist_y

        super_state[10:26] = np.array(state_target).reshape(-1)
        

        super_state[26:30] = np.array(head_direction)
        super_state[30:34] = np.array(tail_direction)

        # snake_length = len(snake.snake()) / ((self.max_w * self.max_w) - 1)
        # super_state[34] = snake_length

        # other snake information
        super_state[34:38] = other_snake_head_direction
        super_state[38:] = other_tail_direction

        return super_state
