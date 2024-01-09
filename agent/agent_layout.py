import numpy as np
# contain a class having the basic functionality of an agent in a markov environment

class Agent_layout:
    def __init__(self, action_space:int, obs_space:int ,buffer_size:int=100000, lr:float=0.005, exploration:int=100, use_next_state:bool=False) -> None:
        self.buffer_size = buffer_size # steps per episodes
        self.action_space = action_space
        self.obs_space = obs_space

        # for storirng the observation
        self.states_memory = np.zeros((self.buffer_size, *obs_space), dtype=np.float32)
        self.action_memory = np.zeros((self.buffer_size), dtype=np.int8)
        self.rewards_memory = np.zeros((self.buffer_size), dtype=np.float32)
        self.is_done_memory = np.zeros((self.buffer_size), dtype=np.bool8)
        
        self.use_next_state = use_next_state
        self.next_state_memory = None
        if use_next_state :
            self.next_state_memory = np.zeros((self.buffer_size, *obs_space), dtype=np.float16)
        
        self.index = 0 # index to keep track
        self.actual_sequence = [] # store the steps infos in temporary buffer

        self.exploration_rate = exploration # amount of exploration
        self.num_episodes = 0


        self.policy = None # model to perform the policy operations

        # for automatically report at the end of episodes
        self.auto_report = True

    
    def react(self, state) :
        # take an action based on the given state
        # exploration vs exploitation
        epsilon = self.exploratopn_rate - self.num_episodes
        if np.random.randint(low=0, high=100) < epsilon:
            action = np.random.randint(0, self.action_space)

        else :
            action = self.custom_policy_action(state)

        return action
    
    def custom_policy_action(self, state):
        pass # define this function in agent

    
    def observe(self, state:np.ndarray=None, reward:float=None ,is_done:bool=False, next_state=None, store:bool=False) :
        if store :
            # when the action has already been performed, store the infos
            self.actual_sequence.append(reward)
            self.actual_sequence.append(is_done)

            if self.use_next_state :
                self.next_state_memory[self.index] = next_state                
            
            self.store_observation(short_train=is_done)

        else :
            self.actual_sequence = [] # reset the sequence

            self.actual_sequence.append(state)
            action = self.react(state) # take an action

            self.actual_sequence.append(action)

            return action # return the action
        

    def store_observation(self, short_train=False):
        state, action, reward, is_done = self.actual_sequence

       
        # storing the values in their arrays
        self.states_memory[self.index] = state
        self.action_memory[self.index] = action
        self.rewards_memory[self.index] = reward
        self.is_done_memory[self.index] = is_done

      


        self.index += 1
        if self.auto_report :
            if not short_train :
                if self.index >= self.buffer_size :
                    self.report()

            else :
                self.report()

        if self.index >= self.buffer_size:
            # perform the training on the policy
            self.report()


    def report(self, trajectory:np.ndarray=None):
        if trajectory :
            self.policy.update(*trajectory)


        elif self.index <= self.buffer_size  and self.index > 1:
            # meaning the episodes collections was stopped abruptly            

            train_states_memory = self.states_memory[:self.index]
            train_action_memory = self.action_memory[:self.index]
            train_rewards_memory = self.rewards_memory[:self.index]
            train_is_done_memory = self.is_done_memory[:self.index]
            


        # updating the model
            replay_buffer = [train_states_memory, train_action_memory, train_rewards_memory, train_is_done_memory]
            if self.use_next_state : 
                train_next_state_memory = self.next_state_memory[:self.index]
                replay_buffer.append(train_next_state_memory)
        # if self.use_next_state :
        #     replay_buffer.append(np.array(train_next_state_memory))


            if self.index != 0 :
                self.policy.update(*replay_buffer)

        # reset everything if we reach limit
        if self.index >= self.buffer_size :
           self.clear_memory()
        
        self.num_episodes += 1 # for exploration

    def clear_memory(self):
        self.states_memory = np.zeros((self.buffer_size, *self.obs_space), dtype=np.float16)
        self.action_memory = np.zeros((self.buffer_size), dtype=np.int8)
        self.rewards_memory = np.zeros((self.buffer_size), dtype=np.float32)
        # self.is_done_memory = [0 for _ in range(self.buffer_size)]
        self.is_done_memory = np.zeros((self.buffer_size), dtype=np.bool8)

        
        if self.use_next_state :
            self.next_state_memory = np.zeros((self.buffer_size, *self.obs_space), dtype=np.float16)
        
        self.index = 0


    def save_model(self, name:str):
        self.policy.save_model(name)
        
                