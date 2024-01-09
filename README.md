# Reinforcement_Learning

## Requirements :

Basic requirements
``` python
python >= 3.10
numpy >= 1.24.3
gymnasium == 0.29.1
pygame == 2.5.2
```

In case you want to use the  torch or tensorflow agents make sure to get the libraries
```
tensorflow == 2.13.0
torch
```

## Usage :

to initialize an agent, first generate a neural network that will act as its brain, you can use my DeepLearningNumpy project at this link
to create a custom network : https://github.com/Bouscout/Deep_learning_numpy

or you can simply create a model from the module neural_v3
``` python
from neural_v3.network import network
from neural_v3.activations import softmax, linear
from agent.agent_numpy.policy_gradient_PPO import Agent_PPO

BUFFER = 1000
obs_space = 10
action_space = 4

brain = network([obs_space, 64, action_space], "relu", learning_rate=1e-4, optimizer="adam")
brain.activ_layers[-1] = softmax()

agent = Agent_PPO(action_space, obs_space, BUFFER, use_next_state=True)
agent.policy.model = brain
```

if your algorithm requires a baseline and a advantage function you can define them from their module
```python
from agent.agent_numpy.advantages_func import GAE_estimations, discounted_rerward, td_estimations, monte_carlo_estimate

baseline_model = network([obs_space, 64, 1], "relu", learning_rate=1e-4, optimizer="adam")
baseline_model.activ_layers[-1] = linear()

reward_function = GAE_estimations()
reward_function.baseline = baseline_model

agent.policy.advantage_func = reward_function

```


training loop usage, the function observe will provide an action, but when call with the argument store=True, it will store the last observations inside the replay buffer

```python
total_steps = 100_000
interval = 500

for step in range(1, total_steps) :
    action = agent.observe(state)

    next_state, reward, done, truncated, _ = env.step(action)

    # save observation
    agent.observe(next_state=next_state, reward=reward, is_done=done or truncated, store=True)

    state = next_state

    # training step
    if step % interval == 0 :
        agent.report()
        agent.clear_memory()

    
    if done or truncated :
        state = env.reset()
```


if you already have a saved model and just to see the agent interaction with the environment

```python
for step in range(1, total_steps) :
    action = agent.observe(state)

    next_state, reward, done, truncated, _ = env.step(action)

    state = next_state

    if done or truncated :
        state = env.reset()

```
```