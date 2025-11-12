import gymnasium as gym

### WRITE YOUR CODE HERE
env = gym.make('Taxi-v3', render_mode='ansi')
state, info = env.reset()
print(env.render())
print(env.action_space)
print(env.observation_space)

obs, reward, terminated, truncated, _ = env.step(0)
print('Observation:', obs)
print('Reward:', reward)
print(env.render())
env.close()