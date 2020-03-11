#!/usr/bin/env python3
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10000

SHOW_EVERY = 500

epsilon = 0
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

discrete_os_size = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/discrete_os_size


q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size)+[env.action_space.n])

ep_rewards = []
aggr_ep_rewards = {'ep':[], 'avg':[], 'min':[], 'max':[]}


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for e in range(EPISODES):

    episode_reward = 0

    if e % SHOW_EVERY == 0:
        render = True
        print(e)
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())

    done = False
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        episode_reward += reward

        #will use this in our Q formula
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state+(action, )]


            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state+(action, )] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0
            print(f"we made it on episode{e}")

        discrete_state = new_discrete_state
    
    if END_EPSILON_DECAYING >= e >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    
    ep_rewards.append(episode_reward)

    if not e % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(e)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Episode: {e} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")

env.close()

plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["avg"], Label="avg")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["min"], Label="min")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["max"], Label="max")
plt.grid(True)
plt.legend(loc=4)
plt.show()
