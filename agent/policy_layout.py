# contain the basic shape for the all the logic for the different types of models used by the agents
import numpy as np
from neural_v3.network import network
import pickle


class Policy_Layout:
    def __init__(self, lr:float=0.005, input_size:int=None, output_size:int=None ) -> None:
        self.lr = lr # learning rate

        self.gamma = 0.99 # discount rate

        # self.model = network(neurons=structure, activation='relu', learning_rate=self.lr)
        # if create a model here and declare a new one later, the address of the numpy arrays might be causing some issues, must be checked
        self.model = None

        self.mini_batch_size = 8

        self.batch_gradient = 0

        self.MAX_VARIANCE = 30 # maximum amount of variance allower in gradient calculation

        # a custom loss function, should always take a reward and a discount factor as parameters
        self.custom_reward_func = None

        self.early_stop = False

    def update(self, *replay_buffer) :
        # unpacking the trajectory
        # if len(replay_buffer) == 5 :
        #     states, actions, rewards, is_done_values, next_states = replay_buffer
        # else :
        #     states, actions, rewards, is_done_values = replay_buffer
        #     next_states = None

        # # discounted_rewards = self.reward_function(rewards)
        # discounted_rewards = rewards

        index = 0
        batch_index = self.mini_batch_size

        # rewards /= np.std(rewards) or 1

        while batch_index < len(replay_buffer[0]) :
            trajectory = []
            for dataset in replay_buffer :
                trajectory.append(dataset[index:batch_index])

            # batch_state = states[index:batch_index]
            # batch_action = actions[index:batch_index]
            # batch_rewards = discounted_rewards[index:batch_index]
            # batch_is_done = is_done_values[index:batch_index]

            # if hasattr(next_states, '__len__'):
            #     batch_next_state = next_states[index:batch_index]
            # else : 
            #     batch_next_state = None 

            # in case a condition has been met to stop the training
            if self.early_stop : 
                return

            # self.loss_operations(batch_state, batch_action, batch_rewards, batch_is_done, batch_next_state)
            self.loss_operations(trajectory)

            index = batch_index
            batch_index += self.mini_batch_size
        
        else :
            # in case there are elements left
            # batch_state = states[index:]
            # batch_action = actions[index:]
            # batch_rewards = discounted_rewards[index:]
            # batch_is_done = is_done_values[index:]

            # if hasattr(next_states, '__len__'):
            #     batch_next_state = next_states[index:batch_index]
            # else : 
            #     batch_next_state = None
            trajectory = [] 
            for dataset in replay_buffer :
                trajectory.append(dataset[index:])

            self.loss_operations(trajectory)

            # self.loss_operations(batch_state, batch_action, batch_rewards, batch_is_done, batch_next_state)

            index = batch_index
            batch_index += self.mini_batch_size

    def loss_operations(self, *batch_trajectory):
        # perform gradient operation here
        # customize this function according to the policy used
        return
    
    def save_model(self, path:str):
        try :
            with open(path + '.pickle', 'wb') as fichier :
                pickle.dump(self.model, fichier)
            print('model saved at : ', path+'.pickle')

        except :
            print("couldn't save the model")

            
    