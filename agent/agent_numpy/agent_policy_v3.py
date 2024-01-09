from agent.agent_layout import Agent_layout
from agent.agent_numpy.policy_v2 import Policy_v2
from neural_v3.activations import softmax
import numpy as np


class Agent_policy_v3(Agent_layout):
    def __init__(self, action_space: int, obs_space: int, num_steps: int = 10000,lr:float=0.005 , e_r: int = 100, use_next_state=False) -> None:
        super().__init__(action_space=action_space, obs_space=obs_space, buffer_size=num_steps, exploration=e_r, use_next_state=use_next_state)
        self.policy = Policy_v2(output_size=action_space, input_size=obs_space, lr=lr)
        self.softmax = softmax()

    def react(self, state):
        if len(state) != 1 :
            state = state.reshape(1, -1)
        logits = self.policy.model.predict(state)
        # probs = self.policy.distribution.forward_propagation(logits=logits)[0]
        probs = logits[0]

        action = np.random.choice(self.action_space, p=probs)
        return action

    def custom_policy_action(self, state):
        # get the probability distrubition
        logits = self.policy.model.predict(state)

        prob = self.softmax.forward_propagation(logits)
        prob = prob[0]

        # select an action based on the probability
        action = np.random.choice(np.arange(self.action_space), p=prob)
        return action