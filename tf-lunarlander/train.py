import gym
from dqn import DeepQNetwork, Agent
from utils import plotLearning
import numpy as np
from gym import wrappers
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
#    env = gym.make('LunarLander-v2')

    env = gym.make('BattleZone-ram-v0')
    observation = env.reset()
    action = env.action_space.sample()
    print(observation)
    print(action)
    lr = 0.0001
    n_games = 500
    agent = Agent(gamma=0.99, epsilon=1, alpha=lr, input_dims=[128],
                  n_actions=18, mem_size=1000000, n_games=n_games,
                  batch_size=64)

    #load_checkpoint = True
    #if load_checkpoint:
    #    agent.load_models()

    alpha = 'alpha' + str(lr)#.split('.')[1]

    filename = '0-lunar-lander-256x256-' + alpha + '-bs64-adam-faster_decay.png'
    scores = []
    eps_history = []

    score = 0
    #env = wrappers.Monitor(env, "tmp/lunar-lander-3",
    #                         video_callable=lambda episode_id: True, force=True)

    for i in range(n_games):
        done = False
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i-10):(i+1)])
            print('episode: ', i,'score: ', score,
                 ' average score %.3f' % avg_score,
                'epsilon %.3f' % agent.epsilon)
            agent.save_models()
        else:
            print('episode: ', i,'score: ', score)

        observation = env.reset()
        observation = observation / 255
        score = 0
        while not done:
#            env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation_ = observation_ / 255
            score += reward
            agent.store_transition(observation, action,
                                   reward, observation_, int(done))
            observation = observation_
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)
