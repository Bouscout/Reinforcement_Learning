import torch
import torch.nn as nn
import numpy as np
from agent.agent_layout import Agent_layout
from agent.policy_layout import Policy_Layout
from agent.agent_numpy.advantages_func import discounted_rerward
from neural_v3.loss_functions import derivative_MSE

class Deep_Q_network(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()

        self.device = torch.device('cpu')
        if(torch.cuda.is_available()): 
            self.device = torch.device('cuda:0') 
            torch.cuda.empty_cache()
            print("Device : " + str(torch.cuda.get_device_name(self.device)))
        else:
            print("Device : cpu")
        print("============================================================================================")

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        self.model.to(self.device)

        self.target = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        self.target.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)

class Deep_Q_N_Policy(Policy_Layout) :
    def __init__(self, lr: float = 0.005, input_size: int = None, output_size: int = None) -> None:
        super().__init__(lr, input_size, output_size)
        self.q_network = Deep_Q_network(input_size, output_size)

        self.model = self.q_network.model
        self.target = self.q_network.target

        # hyperparameters
        self.gamma = 0.99
        self.update_index = 0
        self.update_frequency = 500 # frequency to update target model
        self.mini_batch_size = 64
        self.epochs = 2

        self.MSE = nn.MSELoss()

        self.advantage_func = discounted_rerward(normalize=False)

    def update(self, *replay_buffer):
        # reshaping everything to tensor
        states, actions, rewards, dones, next_states = replay_buffer
        # self.mini_batch_size = len(states)
        # rewards = self.advantage_func.advantages(rewards)

        states = torch.tensor(states, dtype=torch.float32).to(self.q_network.device)
        actions = torch.tensor(actions.reshape(-1), dtype=torch.int32).to(self.q_network.device)
        rewards = torch.tensor(rewards.reshape(-1, 1), dtype=torch.float32).to(self.q_network.device)
        # dones = torch.tensor(dones, dtype=torch.bool).to(self.q_network.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.q_network.device)


        for _ in range(self.epochs) :
            super().update(states, actions, rewards, next_states, dones)

        if self.update_index >= self.update_frequency :
            self.update_target()
            self.update_index = 0
        else : self.update_index += 1

        
        
    def loss_operations(self, batch_trajectory):
        states, actions, rewards, next_states, dones = batch_trajectory

        q_values = self.model(states)

        with torch.no_grad():
            mask = 1 - dones.reshape(-1, 1)
            q_values_next = self.target(next_states)

            # bellman's equation for finding q targets
            max_q_values = torch.max(q_values_next, dim=1).values.unsqueeze(1).detach()
            q_target = rewards + (self.gamma * max_q_values * torch.tensor(mask).to(self.q_network.device))

            # apply gradient to only concerned values
            targets = q_values.clone()

            # this will ensure that the other gradient except the converned action are zero
            targets[torch.arange(targets.size(0)), actions] = q_target.squeeze().to(torch.float32)

            values_test = q_values.cpu().numpy()
            targets_test = targets.cpu().numpy()

            grad = derivative_MSE(values_test, targets_test)
            grad = np.mean(grad, axis=0)

        loss = self.MSE(q_values, targets)

        # taking the gradient step
        self.q_network.optimizer.zero_grad()
        loss.backward()
        self.q_network.optimizer.step()

        grad_model = self.model[-1].bias.grad
        print

    # update target network with parameters of actual model
    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())


class Agent_DeepQNetwork_torch(Agent_layout):
    def __init__(self, action_space: int, obs_space: int, buffer_size: int = 100000, lr: float = 0.005, exploration: int = 100, use_next_state: bool = False) -> None:
        super().__init__(action_space, obs_space, buffer_size, lr, exploration, use_next_state)
        
        self.policy = Deep_Q_N_Policy(lr, obs_space, action_space)
        self.use_next_state = True

        self.exploration_rate = 100
    
    def react(self, state):
        # epsilon greedy
        epsilon = self.exploration_rate - self.num_episodes
        if np.random.randint(low=0, high=50) < epsilon :
            action = np.random.randint(low=0, high=self.action_space)

        else : 
            # use policy to get action
            action = self.custom_policy_action(state)

        return action
    
    def report(self, trajectory= None):

        super().report(trajectory)
        # self.clear_memory()
        return    

    def custom_policy_action(self, state):
        with torch.no_grad() :
            state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(self.policy.q_network.device)
            q_values = self.policy.model(state_tensor).cpu().detach().numpy()

            q_values = q_values[0]

            action = np.argmax(q_values)

            return action