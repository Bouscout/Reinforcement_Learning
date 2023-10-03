# this file will contain all the different advantages functions for RL to be shaping up the rewards
import numpy as np
from neural_v3.loss_functions import derivative_MSE, MSE

class discounted_rerward():
    def __init__(self, gamma=0.99, normalize=True, force_normal=True) -> None:
        self.gamma = gamma

        self.normalize = normalize
        self.force_normal = force_normal

    def advantages(self, rewards:np.ndarray, *args) :
        rewards = np.squeeze(rewards)
        discounted_rerward = np.zeros_like(rewards)

        cumilative_reward = 0

        for t in reversed(range(len(rewards))) :
            discounted_rerward[t] = rewards[t] + (self.gamma * cumilative_reward)
            cumilative_reward = discounted_rerward[t]

        if self.normalize :

            # in some cases substracting the mean brings unexpected interpretation of the rewards
            if self.force_normal :
                discounted_rerward -= np.mean(discounted_rerward)
            else : 
                if np.max(discounted_rerward) > 0 : discounted_rerward -= np.mean(discounted_rerward)

            discounted_rerward /= np.std(discounted_rerward)

        return discounted_rerward[:, None]

class monte_carlo_estimate():
    def __init__(self, gamma:float=0.99, baseline:object=None) -> None:
            self.gamma = gamma
            self.baseline = baseline

    def advantages(self, rewards, states, dones, normalize=True):
        estimations = self.baseline(states)
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        mask = 1 - dones

        discount_reward = np.zeros_like(rewards)
        cumilative_reward = 0
        
        for t in reversed(range(len(dones))):
            discount_reward[t] = rewards[t] + (self.gamma * mask[t] * cumilative_reward)
            cumilative_reward = discount_reward[t]

        # normalize reward
        discount_reward -= np.mean(discount_reward)
        discount_reward /= np.std(discount_reward) + 1e-10

        advantages = discount_reward - estimations

        return advantages, discount_reward
    
    def train_advantage_func(self, states, targets, batch_size=None) :
        # func to take one gradient step
        def train_model(index, batch_index):
            batch_states = states[index:batch_index]
            returns = targets[index:batch_index]

            estim = self.baseline(batch_states)
            gradient = derivative_MSE(estim, returns)

            gradient *= 0.5

            self.baseline.adjust(gradient, average=True)

        # dividing it into batches
        limit = len(states)
        batch_index = batch_size if batch_size else limit
        index = 0

        while batch_index < limit :
            train_model(index, batch_index)
            index = batch_index
            batch_index += batch_size
        
        else :
            batch_index = None
            train_model(index, batch_index)
            

class value_function_Bellman():
    def __init__(self, gamma:float=0.99, baseline:object=None) -> None:
        self.gamma = gamma
        self.baseline = baseline

    def advantages(self, rewards, states, dones, normalize=True):
        # the only unknow in this would be the estimations
        # we will  be using the bellman's equation to find the advantage
        estimations = self.baseline.predict(states)

        next_rewards = np.zeros_like(rewards)
        next_rewards[:-1] = rewards[1:]

        mask = 1 - np.array(dones)
        mask = mask

        # bellman's equation
        return_values = rewards + (self.gamma * next_rewards * mask)
        
        advantages = return_values[:, None] - estimations

        gradient = derivative_MSE(y_predict=estimations, y_true=return_values[:, None])

        self.baseline.adjust(gradient) # fitting to the returns

        if normalize :
            advantages -= np.mean(advantages)
            advantages /= np.std(advantages)

        return advantages 

# we will use the time difference error
class td_estimations():
    def __init__(self, gamma:float=0.99, baseline:object=None) -> None:
        self.gamma = gamma
        self.baseline = baseline

    def advantages(self, rewards, states, dones, normalize=False):
        estimations = self.baseline.predict(states)
        
        # parsing the next states for derivative calculation later
        nx_states = np.zeros_like(states)
        nx_states[:-1] = states[1:]
        nx_states = nx_states[:-1]

        

        # we parse the next estimations from the estimations without the need of using next states
        next_estimations = np.zeros_like(estimations)
        next_estimations[:-1] = estimations[1:]

        mask = 1 - np.array(dones)
        mask = mask[:, None]

        next_estimations = mask * next_estimations

        #td error = reward + (gamma * nx_estim) - estim
        td_error = rewards[:, None] + (self.gamma * next_estimations) - estimations

        # let's find the partial derivatives with respect to both predictions
        # we assume loss = td_error ** 2

        # using the chain rule
        # the estimations derivative
        partial_deriv_estim = 2 * td_error * -1
        self.baseline.adjust(partial_deriv_estim)

        # next estimations derivative
        # assuming that we even have a next state
        if len(nx_states) > 1 :
            partial_deriv_next_estim = 2 * td_error * self.gamma * mask
            partial_deriv_next_estim = partial_deriv_next_estim[:-1]

            # derivative with respect to next states
            self.baseline.predict(nx_states)
            self.baseline.adjust(partial_deriv_next_estim)

        if normalize :
            td_error -= np.mean(td_error)
            td_error /= np.std(td_error)
        return td_error
    
class GAE_estimations:
    def __init__(self, gamma=0.99, baseline=None, _lambda=0.95 ) -> None:
        self.gamma = gamma
        self._lambda = _lambda
        self.baseline = baseline

    def advantages(self, rewards, states, next_states, dones, normalize=True):
        values = self.baseline(states)
        next_values = self.baseline(next_states)

        # in order to filter the done states
        mask = 1 - np.array(dones)
        mask = mask[:, None]

        next_values = next_values * mask

        td_error = rewards[:, None] + (self.gamma * next_values) - values

        advantages = np.zeros_like(td_error)
        gae = 0

        for t in reversed(range(len(td_error))) :
            gae = td_error[t] + (self.gamma * self._lambda * mask[t] * gae)
            advantages[t] = gae

        returns = advantages + values

        if normalize :
            advantages -= np.mean(advantages)
            advantages /= np.std(advantages)

        return advantages, returns

    def train_advantage_func(self, states, targets_batch, old_values_batch, batch_size, loss_clip=0.2):
        
        # function to perform a gradient step 
        def train_model(index, batch_index):
            batch_states = states[index:batch_index]
            returns = targets_batch[index:batch_index]
            old_values = old_values_batch[index:batch_index]

            estimations = self.baseline(batch_states)

            clipped_estim = old_values + np.clip(estimations - old_values, -loss_clip, loss_clip)

            gradient = derivative_MSE(y_predict=estimations, y_true=returns)
            
            # we will apply the clipping to the gradient
            for i in range(len(gradient)) :
                difference = estimations[i] - old_values[i]
                if difference > loss_clip or difference < -loss_clip :
                    if (clipped_estim[i] - returns[i])**2 >= (estimations[i] - returns[i])**2 :
                        gradient[i] = 0

            gradient *= 0.25

            self.baseline.adjust(gradient, average=True)

        # dividing the batch in minibatch of size batch_size
        limit = len(states)

        index = 0
        batch_index = batch_size if batch_size else limit
        while batch_index < limit :
            train_model(index=index, batch_index=batch_index)

            index = batch_index
            batch_index += batch_size

        else :
            # perform training on last remaining elements
            train_model(index=index, batch_index=None)            




        
            




