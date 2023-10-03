import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agent.agent_layout import Agent_layout
from agent.policy_layout import Policy_Layout
from neural_v3.loss_functions import derivative_MSE, MSE
from agent.agent_numpy.advantages_func import GAE_estimations


class conv_network(nn.Module):
    def __init__(self, input_size, output_size, lr=0.0025) -> None:
        super(conv_network, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool_1 = nn.MaxPool2d(1, 1)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fcl_1 = nn.Linear(16 * 1 * 1, 64)
        self.fc_2 = nn.Linear(64, output_size)
        self.fc_3 = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.pool_1(F.relu(self.conv1(x)))
        x = self.pool_1(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 1 * 1)
        x = F.relu(self.fcl_1(x))
        x = self.fc_3(self.fc_2(x))

        return x    

class Actor_Critic():
    def __init__(self, input_size, output_size, learning_rate, conv_layer=False) -> None:
        super().__init__()

        self.device = torch.device('cpu')
        if(torch.cuda.is_available()): 
            self.device = torch.device('cuda:0') 
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(self.device)))
        else:
            print("Device set to : cpu")
        print("============================================================================================")

        if conv_layer :
            self.model = conv_network(input_size, output_size)

            # critic network
            self.baseline = conv_network(input_size, 1)
            self.baseline.fc_3 = nn.Linear(1, 1)
        
        else :
            self.model = nn.Sequential(
                nn.Linear(*input_size, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, output_size),
                nn.Softmax(dim=-1)
            )


            self.baseline = nn.Sequential(
                nn.Linear(*input_size, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )


        # movinf models to appropriate device
        self.model.to(self.device)
        self.baseline.to(self.device)
        
        self.alpha = learning_rate

        self.optimizer = torch.optim.Adam([
            {"params" : self.model.parameters(), "lr": 0.0003},
            {"params" : self.baseline.parameters(), "lr" : 0.001},
        ])





class PPO_torch(Policy_Layout):
    def __init__(self, lr: float = 0.005, input_size: int = None, output_size: int = None) -> None:
        super().__init__(lr, input_size, output_size)

        self.actor_critic = Actor_Critic(input_size, output_size, lr)

        self.model = self.actor_critic.model
        self.baseline = self.actor_critic.baseline

        # hyperparameters
        self.epochs = 10
        self.mini_batch_size = 16
        self.ppo_clip = 0.2
        self.gamma = 0.99
        self._lamda = 0.95

        self.kl_divergence = 0.015

        self.mse = nn.MSELoss()

    def monte_carlo_estimate(self, rewards, states, dones, normalize=True):
        mask = 1 - dones

        cumilative_reward = 0
        discounted_reward = np.zeros_like(rewards)

        for t in reversed(range(len(dones))) :
            discounted_reward[t] = rewards[t] + (self.gamma * cumilative_reward * mask[t])  
            cumilative_reward = discounted_reward[t]


        target = torch.tensor(discounted_reward, dtype=torch.float32).to(self.actor_critic.device)

        if normalize :
            target -= target.mean()
            target /= target.std()

        values = self.baseline(states)

        advantages = target.detach() - values.detach()


        return torch.squeeze(advantages), target
    
   
        

    def GAE_estimations(self, rewards, states, next_states, dones, normalize=True):
        with torch.no_grad() :
            # find td error
            values_tensor = self.baseline(states)
            next_values_tensor = self.baseline(next_states)

            # converting estimations to numpy for gaes calculation
            values = values_tensor.cpu().numpy()
            next_values = next_values_tensor.cpu().numpy()
            mask = 1 - dones

            td_error = rewards + (self.gamma * next_values * mask) - values

            # find gaes and return values
            advantages = np.zeros_like(td_error)
            gaes = 0

            for t in reversed(range(len(td_error))):
                gaes = td_error[t] + (self.gamma * self._lamda * mask[t] * gaes)
                advantages[t] = gaes

            target = advantages + values
            
            if normalize :
                advantages -= np.mean(advantages)
                advantages /= np.std(advantages)

            return advantages, target, values_tensor
        
    def gae_train(self, states, targets, old_values) :
        # find critic clipped loss
        values = self.baseline(states)

        values_clipped = old_values + torch.clamp(values - old_values, -self.ppo_clip, self.ppo_clip)

        critic_loss_clipped = (torch.squeeze(values_clipped) - targets)**2
        critic_loss = (torch.squeeze(values) - targets)**2

        critic_max_loss = torch.max(critic_loss, critic_loss_clipped)
        
        critic_loss_final = 0.5 * critic_max_loss

        return critic_loss_final.mean()

    def get_log_probs_and_values(self, states, actions):
        probs = self.model(states)
        dist = torch.distributions.Categorical(probs=probs)

        log_probs = dist.log_prob(actions)

        return log_probs, self.baseline(states), dist.entropy()
        

    def update(self, *replay_buffer):
        states, actions, rewards, dones, next_states = replay_buffer
        
        dones = dones.reshape(-1, 1)
        rewards = rewards.reshape(-1, 1)

        states_tensor = torch.tensor(states, dtype=torch.float32).detach().to(self.actor_critic.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int8).detach().to(self.actor_critic.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(self.actor_critic.device)

        # gae function
        with torch.no_grad() : 
            # advantages, target = self.monte_carlo_estimate(rewards, states_tensor, dones)
            advantages, target, old_values_tensor = self.GAE_estimations(rewards, states_tensor, next_states_tensor, dones)
            advantages = torch.squeeze(torch.tensor(advantages)).to(self.actor_critic.device)
            target = torch.squeeze(torch.tensor(target)).to(self.actor_critic.device)
       
        with torch.no_grad():
            old_probs_tensor, *_ = self.get_log_probs_and_values(states_tensor, actions_tensor)
            old_probs_tensor = old_probs_tensor.detach()

        for _ in range(self.epochs) :
            super().update(states_tensor, actions_tensor, advantages, old_probs_tensor, target, old_values_tensor)
            # self.loss_operations(states_tensor, actions_tensor, advantages, old_probs_tensor, target, old_values_tensor)
        
    def loss_operations(self, batch_trajectory):
        states, actions, advantages, old_log_probs, target_values, old_values = batch_trajectory

        new_log_probs, new_values, entropy = self.get_log_probs_and_values(states, actions)
        log_ratio = new_log_probs - old_log_probs

        ratio = log_ratio.exp()

        loss_1 = ratio * advantages
        loss_2 = torch.clamp(ratio, 1-self.ppo_clip, 1+self.ppo_clip) * advantages

        ppo_loss = -torch.mean(torch.min(loss_1, loss_2))

        # critic loss
        # critic_loss = self.mse(new_values, target_values.detach()) if monte carlo estimate
        critic_loss = self.gae_train(states, target_values, old_values)

        # total_loss = ppo_loss + (0.5 * critic_loss) - 0.001 * torch.mean(entropy)
        total_loss = ppo_loss + (0.5 * critic_loss) 

        

        self.actor_critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor_critic.optimizer.step()

      


# agent implementation
class Agent_PPO_torch(Agent_layout):
    def __init__(self, action_space: int, obs_space: int, buffer_size: int = 100000, lr: float = 0.005, exploration: int = 100, use_next_state: bool = False) -> None:
        super().__init__(action_space, obs_space, buffer_size, lr, exploration, use_next_state)

        self.policy = PPO_torch(lr, obs_space, action_space)

    def report(self, trajectory=None):
        if trajectory:
            self.policy.update(*trajectory)
            return
        
        train_states_memory = self.states_memory[:self.index]
        train_action_memory = self.action_memory[:self.index]
        train_rewards_memory = self.rewards_memory[:self.index]
        train_is_done_memory = self.is_done_memory[:self.index]
        train_next_states_memory = self.next_state_memory[:self.index]

        replay_buffer = [train_states_memory, train_action_memory, train_rewards_memory, train_is_done_memory, train_next_states_memory]            

        self.policy.update(*replay_buffer)

        if self.index >= self.buffer_size :
            self.clear_memory()

    def react(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state[None, :], dtype=torch.float32).to(self.policy.actor_critic.device)

            probs = self.policy.model(state_tensor)
            
            dist = torch.distributions.Categorical(probs)

            action = dist.sample()

            return action.item()



