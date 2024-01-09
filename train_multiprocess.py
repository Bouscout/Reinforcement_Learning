import multiprocessing
import pygame as pg
from snake_game.game import Snake_Game
from agent.agent_numpy.policy_gradient_PPO import Agent_PPO
from neural_v3.network import network
from neural_v3.activations import softmax, linear
import numpy as np

def batch_collection(conn, batch_size):
    TARGET_SCORE = 250
    INTERVAL = 10

    # receiving the policy
    agent = conn.recv()

    def train_loop(state, sum_reward):
        action = agent.observe(state)

        next_state, reward, done = env.step(action)

        sum_reward += reward

        if done :
            agent.observe(next_state=next_state, reward=reward, is_done=True, store=True)
            return True, None, sum_reward

        agent.observe(next_state=next_state, reward=reward, is_done=False, store=True)
        state = next_state

        return False, state, sum_reward

    env = Snake_Game()
    episode_steps = 1000
    global_step = 1
    reward_list = []
    going = True
    avg_score = 0

    while going :
        sum_reward = 0
        state = env.reset()

        for _ in range(episode_steps) :
            terminated, state, sum_reward = train_loop(state, sum_reward)
            global_step += 1

            if global_step % batch_size == 0 :
                train_states_memory = agent.states_memory[:agent.index]
                train_action_memory = agent.action_memory[:agent.index]
                train_rewards_memory = agent.rewards_memory[:agent.index]
                train_is_done_memory = agent.is_done_memory[:agent.index]
                train_next_state_memory = agent.next_state_memory[:agent.index]
                
                replay_buffer = [train_states_memory, train_action_memory, train_rewards_memory, train_is_done_memory, train_next_state_memory]

                # send the batch collected
                conn.send(replay_buffer)
                conn.send(avg_score)

                # retrieve the latest version of the policy
                agent = conn.recv()

            if terminated :
                reward_list.append(sum_reward)
                avg_score = sum(reward_list[-INTERVAL:]) / len(reward_list[-INTERVAL:])

                if avg_score >= TARGET_SCORE :
                    agent.save_model("ppo_snake_numpy_average_250")

                break

def command_center(train_sessions):
    # model and agent structure
    action_space = 4
    obs_space = 32
    BUFFER = 10_000

    agent = Agent_PPO(action_space, obs_space, BUFFER, use_next_state=True) 

    structure = [128, 128]
    LR = 0.00025
    LR_CRITIC = 0.00025
    activation = 'tanh'
    optimizer = "adam"

    brain = network([*structure, action_space], activation, learning_rate=LR, optimizer=optimizer)
    brain.activ_layers[-1] = softmax()

    baseline = network([*structure, 1], activation, learning_rate=LR_CRITIC, optimizer=optimizer)
    baseline.activ_layers[-1] = linear()
    
    agent.policy.model = brain
    agent.policy.baseline = baseline
    agent.policy.advantage_func.baseline = baseline

    for a_layer ,c_layer in zip(brain.layers, baseline.layers) :
        a_layer.initialize_relu_weights()
        c_layer.initialize_relu_weights()

    # we will only train over the batch from the individual that got the highest score
    def train_over_batches(scores, trajectories):
        best = np.argmax(scores)
        trajectory = trajectories[best]

        agent.report(trajectory)

    going = True
    while going :
        # sending a copy of the agent to all processes
        scores = []
        trajectories = []
        for conn in train_sessions :
            conn.send(agent)
            
        # receiving their reports
        for conn in train_sessions :
            trajectory = conn.recv()
            score = conn.recv()

            trajectories.append(trajectory)
            scores.append(score)

        train_over_batches(scores, trajectories)

if __name__ == "__main__" :
    num_process = 2
    batch_size = 1000
    pipes = []
    processes = []
    for _ in range(num_process) :
        parent_pipe, child_pipe = multiprocessing.Pipe(duplex=True)
        pipes.append(parent_pipe)
        processes.append(multiprocessing.Process(target=batch_collection, args=(child_pipe, batch_size)))

    handler = multiprocessing.Process(target= command_center, args=(pipes,))

    handler.start()
    for batch_process in processes :
        batch_process.start()

