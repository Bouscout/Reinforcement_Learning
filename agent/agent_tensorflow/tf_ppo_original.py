from ..agent_layout import Agent_layout
from ..policy_layout import Policy_Layout
import tensorflow as tf
import tensorflow_probability as tfp
from keras.optimizers import Adam
import numpy as np

class PPO_policy_tf(Policy_Layout) :
    def __init__(self, lr: float = 0.005, input_size: int = None, output_size: int = None) -> None:
        super().__init__(lr, input_size, output_size)

        #hyperparameters
        self.mini_batch_size = 64
        self.epochs = 10
        self.ppo_deviation = 0.2

        self.gamma = 0.99
        self._lambda = 0.95

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, 'relu'),
            tf.keras.layers.Dense(256, 'relu'),
            tf.keras.layers.Dense(64, 'relu'),
            tf.keras.layers.Dense(output_size, 'softmax'),
        ])

        self.model.compile(optimizer=Adam(lr))

        self.baseline = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(*input_size, )),
            tf.keras.layers.Dense(512, 'relu'),
            tf.keras.layers.Dense(256, 'relu'),
            tf.keras.layers.Dense(64, 'relu'),
            tf.keras.layers.Dense(1, 'linear'),
        ])

        def critic_loss(y_true, y_pred):
            old_values, returns = y_true[:, :1], y_true[:, 1:]
            LOSS_CLIPPING = 0.2
            clipped_value_loss = old_values + tf.clip_by_value(y_pred - old_values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (returns - clipped_value_loss) ** 2
            v_loss2 = (returns - y_pred) ** 2
            
            value_loss = 0.5 * tf.reduce_mean(tf.maximum(v_loss1, v_loss2))
            #value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
            return value_loss
        
        self.baseline.compile(optimizer=Adam(lr), loss=critic_loss)

    def gae_estimations(self, rewards, states, next_st, dones):
        num_steps = len(states)

        rewards = rewards.reshape(-1, 1)

        values = self.baseline(states)
        next_values = self.baseline(next_st)

        mask = 1 - dones
        mask = mask.reshape(-1, 1)

        # td = rewards + (gamma * next_values) - values
        td_error = rewards + (self.gamma * next_values * mask) - values

        advantages = np.zeros_like(td_error)
        gae_estimations = 0

        # finding gae estimations for each time step
        for t in reversed(range(num_steps)):
            gae_estimations = td_error[t] + (self.gamma * self._lambda * mask[t] * gae_estimations)
            advantages[t] = gae_estimations

        
        target = advantages + values

        y_true_pair = np.hstack([values, target])

        self.baseline.fit(states, y_true_pair, batch_size=self.mini_batch_size, epochs=self.epochs, verbose=0)

        return advantages

    
    def update(self, *replay_buffer):
        states, actions, rewards, dones, next_states = replay_buffer

        advantages = self.gae_estimations(rewards, states, next_states, dones)

        old_probs = self.model(states)
        # self.mini_batch_size = self.mini_batch_size

        for _ in range(self.epochs) :
            # this function will dive it in size batch of size self.mini_batch_size
            return super().update(states, actions, advantages, old_probs)
        
    def loss_operations(self, batch_trajectory):
        states, actions, advantages, old_probs = batch_trajectory

        epsilon = 1e-10
        with tf.GradientTape() as Tape :
            probs = self.model(states)

            probs = tf.clip_by_value(probs, epsilon, 1.0)
            old_probs = tf.clip_by_value(probs, epsilon, 1.0)

            prob_distribution = tfp.distributions.Categorical(probs=probs)

            action_probs = prob_distribution.log_prob(actions)

            old_prob_distribution = tfp.distributions.Categorical(probs=old_probs)
            old_action_probs = old_prob_distribution.log_prob(actions)

            ratio = tf.math.exp(action_probs - old_action_probs)

            ppo_loss_1 = ratio * advantages
            ppo_loss_2 = tf.clip_by_value(ratio, 1 - self.ppo_deviation, 1 + self.ppo_deviation)

            ppo_loss_min = tf.minimum(ppo_loss_1, ppo_loss_2)
            
            loss = -tf.reduce_mean(ppo_loss_min)

        gradients = Tape.gradient(loss, self.model.trainable_variables)

        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

class tf_ppo_original_agent(Agent_layout):
    def __init__(self, action_space: int, obs_space: int, buffer_size: int = 100000, lr: float = 0.005, exploration: int = 100, use_next_state: bool = False) -> None:
        super().__init__(action_space, obs_space, buffer_size, lr, exploration, use_next_state)

        self.policy = PPO_policy_tf(lr, obs_space, action_space)

    def react(self, state):
        state = state.reshape(1, -1)
        probs = self.policy.model(state).numpy()[0]

        action = np.random.choice(self.action_space, p=probs)

        return action
