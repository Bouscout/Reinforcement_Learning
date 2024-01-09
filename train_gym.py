import gymnasium as gym
# from agent.agent_torch.torch_ppo_agent import Agent_PPO_torch
from agent.agent_numpy.policy_gradient_PPO import Agent_PPO
from agent.agent_numpy.agent_policy_v3 import Agent_policy_v3
# from agent.agent_torch.torch_dqn import Agent_DeepQNetwork_torch
from agent.agent_numpy.deep_q_network import Agent_DQN
# from agent.tf_ppo_agent import PPO_Agent_please
# from agent.agent_tensorflow.tf_ppo_original import tf_ppo_original_agent
from neural_v3.network import network
from neural_v3.activations import softmax, linear
import numpy as np
import pickle

seed = 10

np.random.seed(seed)

Perso = True # set this to true for numpy agent

INTERVAL = 100
BUFFER = 100_000

BATCH_SIZE = 1000
EPI_STEPS = 1000

TRAIN = False

SAVE = True
LOAD = True
# save_file = "models/lunar_lander.pkl"
save_file = "models/cart_pole.pkl"


TARGET_SCORE = 150

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="human")
# env = gym.make("LunarLander-v2", render_mode="human")

action_space = env.action_space.n
obs_space = env.observation_space.shape[0]
obs_space = (obs_space, ) # making it a tuple

agent = None

agent = Agent_PPO(action_space, obs_space, BUFFER, use_next_state=True)
# if Perso :
# else :
#     agent = Agent_PPO_torch(action_space, obs_space, BUFFER, use_next_state=True)

agent.auto_report = True

if Perso :
    struture = [256, 256]
    activation = "relu"
    optim = "adam"
    LR = 0.00005

    if LOAD :
        with open(save_file, "rb") as f :
            brain = pickle.load(f)

        critic = network([*struture, 1], activation, learning_rate=0.001, optimizer=optim)
        critic.activ_layers[-1] = linear()

    else :
        brain = network([*struture, action_space], activation, learning_rate=LR, optimizer=optim)
        brain.activ_layers[-1] = softmax()

        critic = network([*struture, 1], activation, learning_rate=0.001, optimizer=optim)
        critic.activ_layers[-1] = linear()

        for a_layer, c_layer in zip(brain.layers, critic.layers) :
            a_layer.initialize_relu_weights()
            c_layer.initialize_relu_weights()


    agent.policy.model = brain
    agent.policy.baseline = critic
    agent.policy.advantage_func.baseline = critic

def train_loop():
    global sum_reward, state
    action = agent.observe(state)

    next_state, reward, done, truncated, _ = env.step(action)

    sum_reward += reward

    if done or truncated :
        if TRAIN :
            agent.observe(next_state=next_state, reward=reward, is_done=True, store=True)

        return True

    if TRAIN :
        agent.observe(next_state=next_state, reward=reward, is_done=False, store=True)

    state = next_state

going = True
rewards_list = []
global_step = 1
index = 0
while going :
    state = env.reset()[0]
    sum_reward = 0
    index += 1

    for _ in range(EPI_STEPS) :
        terminated = train_loop()
        global_step += 1

        if global_step % BATCH_SIZE == 0 and TRAIN :
            # train the model
            agent.report()
            agent.clear_memory()

        if terminated :
            rewards_list.append(sum_reward)

            avg_score =sum(rewards_list[-INTERVAL:]) / len(rewards_list[-INTERVAL:])

            sum_reward = round(sum_reward, 2)
            avg_score = round(avg_score, 2)

            print(f"episode:{index}\t score:{sum_reward}\t avg score:{avg_score}")

            if avg_score >= TARGET_SCORE and TRAIN:

                if SAVE :
                    with open(save_file, "wb") as f :
                        pickle.dump(brain, f)
                        print("model saved at ", save_file)

                print(f"reached target average score at {index} episode")
                going = False
            break
        


    