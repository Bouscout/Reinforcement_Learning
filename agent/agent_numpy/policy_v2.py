from agent.policy_layout import Policy_Layout
from neural_v3.activations import softmax, linear
from neural_v3.network import network
from neural_v3.loss_functions import derivative_softmax_cross_entropy, derivative_MSE, MSE
from agent.agent_numpy.advantages_func import td_estimations, monte_carlo_estimate, discounted_rerward, GAE_estimations
import numpy as np

class Policy_v2(Policy_Layout):
    def __init__(self, lr: float = 0.005, input_size: int = None, output_size: int = None) -> None:
        super().__init__(lr, input_size, output_size)
        # declare the different models to use
        self.model = None # policy model
        self.softmax = softmax()

        self.mini_batch_size = 1

        # a baseline model to estimate the A(t) function
        # prevent high variance from situation where positive and negative rewards are present
        self.baseline = None 

        self.actor_critic = None

        # self.advantage_func = td_estimations(baseline=self.baseline)
        # self.advantage_func = value_function(baseline=self.baseline)
        # self.advantage_func = GAE_estimations(baseline=self.baseline)
        self.advantage_func = monte_carlo_estimate(baseline=self.baseline)
        # self.advantage_func = discounted_rerward(normalize=True, force_normal=True)


    # def loss_operations(self, state, action, reward, is_done, next_state):
    def loss_operations(self, batch_trajectory):
        state, action, reward = batch_trajectory
        # find the probability distrubition

        # reward = reward[::-1]

        logits = self.model.predict(state)
        prob = self.softmax.forward_propagation(logits)

        # find the jacobian matrice for the different logits probs
        def find_jacobian():
            softmax_value = np.copy(prob)

            # we know that di/dj = softmax * (1 - softmax) if i = j
            # di/dj = softmax[i] * (-softmax[j]) if i != j
            identity = np.eye(prob.shape[-1]) # we will have the point of intersection in the diagonal
            
            # we can reduce the formula to di/dj = softmax * ((i==j * 1) - softmax)
            jacobian = softmax_value[:, :, None] * (identity - softmax_value[:, None, :])
            return jacobian
        
        softmax_jacobian = find_jacobian()



        # find the loss of the batch
        chosen_actions_probs = prob[np.arange(len(state)), action]

        # we know that has the loss probability gets smaller, it means the probability is close to 1
        # so by minimizing the loss with respect to the log probability we maximixe our rewards for that action probability

        # this gradient represent the magnitude at which the network is wrong with respect to the corresponding 
        # reward and the corresponding allocated probability
        loss = -reward * np.log(chosen_actions_probs)


        # advantage = self.advantage_function(state ,action, reward, is_done ,next_state)         
        advantage = reward         

        # we can also infer that the other actions are affecting the loss by raising the probability towards them
        # so we can derive a gradient loss for those actions even though they are not being executed
        gradient_actions = np.zeros_like(prob)

        
        # cross entropy in case all rewards were positive
        deriv_actions_positive_reward = np.copy(prob)
        deriv_actions_positive_reward[np.arange(len(state)) , action] -= 1


        final_gradient_parameters = np.zeros_like(logits) 

        for index in range(len(state)):
            # loss = -advantage * np.log(action_probability) 
            
            # if advantage[index] >= 0 :
            if True:
                # deriv of loss with respect to the final prediction
                gradient_output = advantage[index]

                # deriv of the actions for passing it to the probability distrubition 
                gradient_actions = deriv_actions_positive_reward[index]

                # we could derive the gradient of the log prob and find the gradient of the softmax
                # but turns out their value become very close the gradient of the actions, so let's just keep that one
                gradient_softmax = gradient_actions

                # chain rule
                gradient_parameters = gradient_output * gradient_softmax 

                # final_gradient_parameters[index] = gradient_parameters
            # elif advantage[index] < 0 :
            #     gradient_output = -advantage[index]

            #     gradient_actions = deriv_actions_positive_reward[index]
            #     action_prob = np.copy(gradient_actions[action[index]])

            #     gradient_actions -= 1
            #     gradient_actions[action[index]] = 1 + action_prob

            #     gradient_parameters = gradient_actions * gradient_output
            
            # loss for negative reward would be : loss = -reward * action_probability
            elif advantage[index] < 0:
               # deriv with respect to output
               gradient_output = -advantage[index]

               # since we are using a softmax for the prob distrubition
            #    jacobian = softmax_jacobian[index]
            #    action_jacobian = np.copy(jacobian[action[index]])

               # increase the other probabilities
            #    jacobian *= -gradient_output

               # decrease the action prob
            #    action_jacobian *= gradient_output
               gradient_prob_action = softmax_jacobian[index, action[index]]

            #    jacobian[action[index]] = action_jacobian

               # using the chain rule
               gradient_parameters = gradient_prob_action * gradient_output
            #    gradient_parameters = np.sum(jacobian, axis=0, keepdims=True)
          
            	
            final_gradient_parameters[index] = gradient_parameters
           
            # gradient_actions[index, action[index]] = prob_gradient

        self.model.adjust(final_gradient_parameters, average=False)    
        


    
    def update(self, *replay_buffer):
        self.mini_batch_size = len(replay_buffer[0])

        states, actions, rewards, dones, next_states = replay_buffer

        # advantages = self.advantage_func.advantages(replay_buffer[2], replay_buffer[0], replay_buffer[3])
        advantages, targets = self.advantage_func.advantages(rewards, states, dones)

        self.advantage_func.train_advantage_func(states, targets)

        # advantages = self.td_error_estimations(replay_buffer[0], replay_buffer[2], replay_buffer[3])

        # advantages = self.discount_reward(replay_buffer[2], True, True)
        
        # replay_buffer[2] = advantages

        # shuffled_index = np.random.permutation(len(replay_buffer[0]))
        # for index in range(len(replay_buffer) ) :
        #     replay_buffer[index] = replay_buffer[index][shuffled_index]


        # return super().update(*replay_buffer)
        return super().update(states, actions, advantages)
    
    