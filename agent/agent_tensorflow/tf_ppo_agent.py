import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Input
import keras.backend as K
import numpy as np
from agent.agent_layout import Agent_layout
from agent.policy_layout import Policy_Layout
import copy


class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space

        # X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        # X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        # X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
       
        X = Dense(64, activation="tanh")(X_input)
        X = Dense(64, activation="tanh")(X)

        output = Dense(self.action_space, activation="softmax")(X)

        self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=lr))

    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        # entropy = -(y_pred * K.log(y_pred + 1e-10))
        # entropy = ENTROPY_LOSS * K.mean(entropy)
        
        # total_loss = actor_loss - entropy
        total_loss = actor_loss 

        return total_loss

    def predict(self, state):
        return self.Actor(state)


class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        old_values = Input(shape=(1,))

        lr = 0.001
        # V = Dense(512, activation="relu", kernel_initializer='he_uniform')(X_input)
        # V = Dense(256, activation="relu", kernel_initializer='he_uniform')(V)
        # V = Dense(64, activation="relu", kernel_initializer='he_uniform')(V)


        V = Dense(64, activation="tanh")(X_input)
        V = Dense(64, activation="tanh")(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=X_input, outputs = value)
        self.Critic.compile(loss=[self.critic_PPO2_loss()], optimizer=optimizer(learning_rate=lr))

    def critic_PPO2_loss(self):
        def loss(y_true, y_pred):
            old_values, returns = y_true[:, :1], y_true[:, 1:]
            LOSS_CLIPPING = 0.2
            clipped_value_loss = old_values + K.clip(y_pred - old_values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (returns - clipped_value_loss) ** 2
            v_loss2 = (returns - y_pred) ** 2
            
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            #value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
            return value_loss
        return loss

    def predict(self, state):
        return self.Critic(state)


class PP0(Policy_Layout) :
    def __init__(self, lr: float = 0.005, input_size: int = None, output_size: int = None) -> None:
        super().__init__(lr, input_size, output_size)
        self.action_space = output_size

        optimizer = Adam

        self.mini_batch_size = 64
        self.epochs= 80

        self.baseline = Critic_Model(input_size, output_size, lr, Adam)

        self.model = Actor_Model(input_size, output_size, lr, optimizer)

    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)

        # gaes = np.zeros_like(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]
            # gaes[t] = deltas[t] + (1 - dones[t]) * gamma * lamda * deltas[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)
    
    def loss_operations_2(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions 
        values = self.baseline.Critic(states)
        next_values = self.baseline.Critic(next_states)

        # Compute discounted rewards and advantages
        #discounted_r = self.discount_rewards(rewards)
        #advantages = np.vstack(discounted_r - values)
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
      
        y_true = np.hstack([advantages, predictions, actions])
        
        old_value_w_target = np.hstack([values, target])
        # training Actor and Critic networks
        a_loss = self.model.Actor.fit(states, y_true, epochs=self.epochs, batch_size=len(states), verbose=0)
        c_loss = self.baseline.Critic.fit(states, old_value_w_target, epochs=self.epochs, batch_size=len(states) ,verbose=0)

    def loss_operations(self, states, actions, rewards, dones, next_states, predictions):
        actions_one_hot = np.zeros((len(actions), self.action_space))
        actions_one_hot[np.arange(len(actions)), actions] = 1

        actions = actions_one_hot

        values = self.baseline.Critic(states)
        next_values = self.baseline.Critic(next_states)

        # Compute discounted rewards and advantages
        #discounted_r = self.discount_rewards(rewards)
        #advantages = np.vstack(discounted_r - values)
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        '''
        pylab.plot(advantages,'.')
        pylab.plot(target,'-')
        ax=pylab.gca()
        ax.grid(True)
        pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
        pylab.show()
        '''
        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom PPO loss function we unpack it
        y_true = np.hstack([advantages, predictions, actions])
        
        old_value_w_target = np.hstack([values, target])
        # training Actor and Critic networks
        a_loss = self.model.Actor.fit(states, y_true, epochs=self.epochs, batch_size=len(states), verbose=0)
        c_loss = self.baseline.Critic.fit(states, old_value_w_target, epochs=self.epochs, batch_size=len(states) ,verbose=0)
        # c_loss = self.Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

    def update(self, *replay_buffer):
        state, action, reward, done, next_state = replay_buffer
        old_predictions = self.model.Actor(state)


        self.mini_batch_size = len(replay_buffer[0])
        self.loss_operations(state, action, reward, done, next_state, old_predictions)
        
       

class PPO_Agent_please(Agent_layout):
    def __init__(self, action_space: int, obs_space: int, buffer_size: int = 100000, lr: float = 0.005, exploration: int = 100, use_next_state: bool = False) -> None:
        super().__init__(action_space, obs_space, buffer_size, lr, exploration, use_next_state)

        self.action_space = action_space
        self.policy = PP0(lr, obs_space, action_space)

    def custom_policy_action(self, state):
        prediction = self.policy.model.predict(state)[0]
        action = np.random.choice(self.action_space, p=prediction)
        action_onehot = np.zeros([self.action_space])
        action_onehot[action] = 1
        return action, action_onehot, prediction
    
    def react(self, state):
        if len(state) != 1 :
            state = state[None, :]

        prediction = self.policy.model.Actor(state).numpy()[0]
        action = np.random.choice(self.action_space, p=prediction)
        return action