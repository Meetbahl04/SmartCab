import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from q_learning_agent import QLearningAgent
def train_agent(agent, env, n_episodes, eval_interval = 100):
    returns = []
    best_avg_return = -np.inf

    for episodes in range(n_episodes):
        obs, _ = env.reset()
        done = False
        n_steps = 0
        total_reward = 0

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            obs = next_obs
            done = terminated or truncated

            n_steps += 1
            total_reward += reward
        
        agent.decay_epsilon()
        #print(f'Episode{episodes}: Length = {n_steps}, Return = {total_reward}')
        returns.append(total_reward)

        if episodes >= eval_interval:
            avg_return = np.mean(returns[episodes - eval_interval : episodes])
            if avg_return > best_avg_return:
                best_avg_return = avg_return
        
        if episodes % eval_interval == 0 and episodes > 0:
            print(f'Episode {episodes}: best average return = {best_avg_return}')
        
    return returns

n_episodes = 70000
learning_rate = 0.5
initial_epsilon = 1.0
final_epsilon = 0
epsilon_decay = (initial_epsilon - final_epsilon)/ (n_episodes/2)

env = gym.make('Taxi-v3')
agent = QLearningAgent(env = env, learning_rate = learning_rate, initial_epsilon = initial_epsilon, epsilon_decay = epsilon_decay, final_epsilon = final_epsilon)

returns = train_agent(agent, env, n_episodes)

def plot_returns(returns, file_name):
    plt.plot(np.arange(len(returns)), returns)
    plt.title('Episode returns')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.savefig(file_name)
    plt.show()

#plot_returns(returns, file_name = 'q_learning_curve.png')

def show_policy(agent, env):
    agent.epsilon = 0
    obs, _ = env.reset()
    env.render()
    done = False
    
    while not done:
        action = agent.get_action(obs)
        next_obs, _, terminated, truncated, _ = env.step(action)
        env.render()
        done = terminated or truncated
        obs = next_obs

env = gym.make('Taxi-v3', render_mode='human')
show_policy(agent, env)
