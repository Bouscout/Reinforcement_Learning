from agent.agent_numpy.policy_gradient_PPO import Agent_PPO
from agent.agent_torch.torch_ppo_agent import Agent_PPO_torch
from neural_v3.network import network
from neural_v3.activations import softmax, linear
from snake_game.double_snake import Snake_Game
import pickle

action_space = 4
obs_space = (42, )
# obs_space = (1, 5, 5)

perso = True

BATCH_SIZE = 250
TARGET_SCORE = 350

BUFFER = 10000

EPI_STEPS = 2000

INTERVAL = 10

env = Snake_Game()

agent = Agent_PPO(action_space, obs_space, BUFFER, use_next_state=True) 
# agent = Agent_PPO_torch(action_space, obs_space, BUFFER, use_next_state=True)
agent.auto_report = False
if perso :
    structure = [256, 128]
    LR = 0.00025
    LR_CRITIC = 0.00025
    activation = 'tanh'
    optimizer = "adam"

    try :
        brain = pickle.load(open("ppo_double_snake.pickle", 'rb'))
        print("checkpoint loaded")
        
        baseline = network([*structure, 1], activation, learning_rate=LR_CRITIC, optimizer=optimizer)
        baseline.activ_layers[-1] = linear()

    except :
        brain = network([*structure, action_space], activation, learning_rate=LR, optimizer=optimizer)
        brain.activ_layers[-1] = softmax()

        baseline = network([*structure, 1], activation, learning_rate=LR_CRITIC, optimizer=optimizer)
        baseline.activ_layers[-1] = linear()
        
        for a_layer ,c_layer in zip(brain.layers, baseline.layers) :
            a_layer.initialize_relu_weights()
            c_layer.initialize_relu_weights()
   
    agent.policy.model = brain
    agent.policy.baseline = baseline
    agent.policy.advantage_func.baseline = baseline



def train_loop():
    global state, sum_reward
    state_1, state_2 = state

    action_1 = agent.observe(state_1)
    action_2 = agent.custom_policy_action(state_2)

    next_state, reward, done = env.step([action_1, action_2])
    nx_state_1, nx_state_2 = next_state

    sum_reward += reward
    if done :
        agent.observe(next_state=nx_state_1, reward=reward/2, is_done=True, store=True)

        agent.observe(state_2)
        agent.actual_sequence[1] = action_2
        agent.observe(next_state=nx_state_2, reward=reward/2, is_done=True, store=True)

        return True

    agent.observe(next_state=nx_state_1, reward=reward, is_done=False, store=True)

    agent.observe(state_2)
    agent.actual_sequence[1] = action_2
    agent.observe(next_state=nx_state_2, reward=reward/2, is_done=False, store=True)

    state = next_state

going = True
index = 0
global_step = 1
reward_list = []
record = 0
avg_reward_record = 0
while going :
    state = env.reset()
    sum_reward = 0

    for _ in range(EPI_STEPS) :
        terminated = train_loop()
        global_step += 1

        if global_step % BATCH_SIZE == 0 :
            agent.report()
            agent.clear_memory()

        if terminated :
            index += 1
            sum_reward = round(sum_reward, 2)
            reward_list.append(sum_reward)


            avg_reward = sum(reward_list[-INTERVAL:]) / len(reward_list[-INTERVAL:])
            avg_reward = round(avg_reward, 2)

            game_score = env.score

            if game_score > record :
                record = game_score

            if avg_reward > avg_reward_record :
                avg_reward_record = avg_reward
                agent.save_model("ppo_double_snake")
            print(f"episode: {index}\t score: {game_score}\t record: {record}\t avg reward:{avg_reward}")

            if avg_reward >= TARGET_SCORE :
                print(f"reached satisfactory results at episode {index}")
                going = False
            break

