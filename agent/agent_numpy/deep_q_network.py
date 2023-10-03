from agent.agent_layout import Agent_layout
from agent.policy_layout import Policy_Layout
from neural_v3.loss_functions import derivative_MSE
from neural_v3.network import network
from neural_v3.activations import linear
import copy
import numpy as np


class Deep_Q_network(Policy_Layout):
    def __init__(self, lr: float = 0.005, input_size: int = None, output_size: int = None) -> None:
        super().__init__(lr, input_size, output_size)
        self.epochs = 5
        self.mini_batch_size = 64
        self.update_frequency = 500

        self.target = None

        structure = [64, 64]
        self.model = network(neurons=[*structure, output_size], activation='relu', learning_rate=lr, optimizer='adam')
        self.model.activ_layers[-1] = linear()

        self.target = copy.deepcopy(self.model)

        self.update_index = 0



    def update(self, *replay_buffer):
        states, actions, rewards, dones, next_states = replay_buffer

        rewards = rewards.reshape(-1, 1)
        for _ in range(self.epochs) :
            
            super().update(states, actions, rewards, next_states, dones)

        if self.update_index >= self.update_frequency :
            self.update_target()
            self.update_index = 0
        else : self.update_index += 1
    
    def loss_operations(self, batch_trajectory):
        states, actions, rewards, next_states, dones = batch_trajectory

        q_values = self.model(states)
        q_next = self.target(next_states)

        mask = 1 - dones.reshape(-1, 1)

        # bellman's quation
        max_q_next = np.max(q_next, axis=1).reshape(-1, 1)
        q_target = rewards + (self.gamma * mask * max_q_next)

        targets = np.copy(q_values)
        
        targets[np.arange(len(targets)), actions] = np.squeeze(q_target)

        gradient = derivative_MSE(y_predict=q_values, y_true=targets)

        self.model.adjust(gradient, average=True)

    def update_target(self):
        params = self.model.get_params()

        self.target.set_params(params)

class Agent_DQN(Agent_layout) :
    def __init__(self, action_space: int, obs_space: int, buffer_size: int = 100000, lr: float = 0.005, exploration: int = 100, use_next_state: bool = False) -> None:
        super().__init__(action_space, obs_space, buffer_size, lr, exploration, use_next_state)

        self.policy = Deep_Q_network(lr, obs_space, action_space)

        self.use_next_state = True
    def report(self, trajectory= None):
        super().report(trajectory)
        # self.clear_memory()

    def react(self, state):
        epsilon = self.exploration_rate - self.num_episodes

        if np.random.randint(0, 50) < epsilon :
            action = np.random.randint(0, self.action_space)

        else :
            action = self.custom_policy_action(state)
        
        return action
    
    def custom_policy_action(self, state):
        q_values = self.policy.model(state.reshape(1, -1))

        action = np.argmax(q_values[0])
        return action