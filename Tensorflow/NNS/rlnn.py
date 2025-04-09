import gym
import numpy as np
import time
import matplotlib.pyplot as plt


def get_average(values):
  return sum(values)/len(values)


env = gym.make('FrozenLake-v1')

states = env.observation_space.n  #num of states
actions = env.action_space.n   #num of actions
q_table = np.zeros([states, actions])

episodes = 1500    #how many times to run the env to train the agent
max_steps = 100 #preventing agent from loopin between states 
learning_rate = 0.81
discount_factor = 0.91
epsilon = 0.9   #staring with 90% chance of pickin random action
render = False
rewards = []

for episode in range(episodes):
  state = env.reset()   #0 at the beginning
  for _ in range(max_steps):
    
    if render:
      env.render()

    if np.random.uniform(0, 1) < epsilon:
      action = env.action_space.sample()  
    else:
      action = np.argmax(q_table[state, :])

    next_state, reward, done, _ = env.step(action)
    q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action])

    state = next_state

    if done: 
      rewards.append(reward)
      epsilon -= 0.001
      break

print(q_table)
print(f"Average reward: {sum(rewards)/len(rewards)}:")

avg_rewards = []
for i in range(0, len(rewards), 100):
  avg_rewards.append(get_average(rewards[i:i+100])) 

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()