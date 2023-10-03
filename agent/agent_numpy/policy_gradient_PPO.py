import numpy as np
from agent.policy_layout import Policy_Layout
from agent.agent_layout import Agent_layout
from agent.agent_numpy.advantages_func import GAE_estimations, discounted_rerward, td_estimations, monte_carlo_estimate
from neural_v3.activations import softmax


class PPO_policy(Policy_Layout):
    def __init__(self, lr: float = 0.005, input_size: int = None, output_size: int = None) -> None:
        super().__init__(lr, input_size, output_size)
        self.action_space = output_size

        self.mini_batch_size = 16
        self.epochs = 10
        self.ppo_deviation = 0.2

        # extra parameters
        self.early_stop = False

        self.distribution = softmax()

        self.baseline = None

        self.advantage_func = GAE_estimations()
        # self.advantage_func = monte_carlo_estimate()
        # self.advantage_func = discounted_rerward(normalize=False)

    def update(self, *replay_buffer):

        states, actions, rewards, dones, next_states, old_probs = replay_buffer
        old_values = self.baseline(states)

        # self.mini_batch_size = len(states)
        advantages, targets = self.advantage_func.advantages(rewards, states, next_states, dones=dones)
        # advantages, targets = self.advantage_func.advantages(rewards, states, dones)
      
      
        old_action_probs = old_probs[np.arange(len(actions)), actions]
        old_action_probs_log = np.log(old_action_probs)
        old_action_probs_log = old_action_probs_log[:, None]


        for _ in range(self.epochs) :
            # this will divide everything in batches of size mini_batch_size and perform self.loss_operations on batches
            super().update(states, actions, advantages, old_action_probs_log)
            self.advantage_func.train_advantage_func(states, targets, old_values, batch_size=self.mini_batch_size)
            # self.advantage_func.train_advantage_func(states, targets, self.mini_batch_size)

      
    def loss_operations(self, batch_trajectory):
        # states, actions, advantages, old_logits = batch_trajectory
        states, actions, advantages, old_action_log_prob = batch_trajectory

        # new prediction
        probs = self.model(states)

        action_prob = probs[np.arange(len(actions)), actions]

        action_prob = action_prob[:, None] # reshapping

        # ppo loss calculation
        ratio = np.exp(np.log(action_prob) - old_action_log_prob)

        loss_1 = ratio * advantages
        loss_2 = np.clip(ratio, 1-self.ppo_deviation, 1+self.ppo_deviation) * advantages

        loss = np.minimum(loss_1, loss_2)

        # we derive the gradient from the loss function
        
        # action prob derivative
        deriv_action_prob = np.copy(probs)
        deriv_action_prob[np.arange(len(states)) , actions] -= 1

      
        # using the chain rule we find the gradient with respect to the network parameters

        # we will be covering the edge cases of ppo
        # we assume that the case where the value is clipped doesn't derive directly from any output from our network so 
        # the gradient will become zero
        # but we still need the minimum loss so we check if we have gradients possibilities are smaller than zero
        for i in range(len(ratio)) :
            ratio_chosen = ratio[i]
            if ratio_chosen < 1 - self.ppo_deviation or ratio_chosen > 1 + self.ppo_deviation:
                if ratio_chosen > 1 + self.ppo_deviation and advantages[i] < 0  :
                    parameter_prob = action_prob[i]

                elif ratio_chosen < 1 - self.ppo_deviation and advantages[i] > 0:
                    parameter_prob = action_prob[i]
                
                else :
                    parameter_prob = 0
                
                action_prob[i] = parameter_prob                
                
        # we use the chain rule to find the gradient with respect to parameters                
        gradient_output = action_prob * np.exp(-old_action_log_prob) * advantages

        gradient_parameters = gradient_output * deriv_action_prob
       
        # backpropagating the gradient
        self.model.adjust(gradient_parameters, average=True)

        
class Agent_PPO(Agent_layout):
    def __init__(self, action_space: int, obs_space: int, buffer_size: int = 100000, lr: float = 0.005, exploration: int = 100, use_next_state: bool = False) -> None:
        super().__init__(action_space, obs_space, buffer_size, lr, exploration, use_next_state)

        self.probs_memory = np.zeros((self.buffer_size, self.action_space), dtype=np.float32)

        self.policy = PPO_policy(lr, obs_space, action_space)

    def report(self, trajectory=None):
    
        train_states_memory = self.states_memory[:self.index]
        train_action_memory = self.action_memory[:self.index]
        train_rewards_memory = self.rewards_memory[:self.index]
        train_is_done_memory = self.is_done_memory[:self.index]
        train_next_states_memory = self.next_state_memory[:self.index]
        train_probs_memory = self.probs_memory[:self.index]

        replay_buffer = [train_states_memory, train_action_memory, train_rewards_memory, train_is_done_memory, train_next_states_memory, train_probs_memory]            

        self.policy.update(*replay_buffer)

        if self.index >= self.buffer_size :
            self.clear_memory()


    def react(self, state):
        if len(state) != 1 :
            state = state.reshape(1, -1)
        logits = self.policy.model.predict(state)
        # probs = self.policy.distribution.forward_propagation(logits=logits)[0]
        probs = logits[0]

        self.probs_memory[self.index] = probs


        action = np.random.choice(self.action_space, p=probs)
        return action



    def custom_policy_action(self, state):
        if len(state) != 1 :
            state = state.reshape(1, -1)
        probs = self.policy.model(state)[0]

        action_index = np.random.choice(self.action_space, p=probs)
        # action_one_hot = np.zeros((self.action_space))
        # action_one_hot[action_index] = 1
        
        return action_index