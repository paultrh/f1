import random


from config import SHOW
import appnope
appnope.nope()
import time
from datetime import datetime

from ai.ddqn import Agent
from models.Car import Car
from models.Map import Map

import torch

import matplotlib.pyplot as plt
import numpy as np
import sys


LOAD = True
SAVE = False
PLAY = False
REPLAY = None

NB_AGENTS = 20


if SHOW or PLAY:
    import pygame
    pygame.init()
    pygame.display.set_caption("F1")

    screen = pygame.display.set_mode((920, 720))

init_pos = (110, 200)


game_engine_status = True

if len(sys.argv) == 3:
    slug = sys.argv[1]
    lr_arg = float(sys.argv[2])
else:
    slug = 'default'
    lr_arg = 0.001


start = time.time()

map = Map('maps/loops.map')
#car1 = Car(init_pos=init_pos, map=map)
#agent = Agent(gamma=0.99, epsilon=0.0, lr=lr_arg, input_dims=[15], batch_size=128, n_actions=3, eps_dec=0.00035)
scores, avg_scores, eps_history = [], [], []
n_games = 2000

agents = []
for i in range(0, NB_AGENTS):
    agents.append(Agent(gamma=0.99, epsilon=0.0, lr=lr_arg, input_dims=[3], batch_size=512, n_actions=3, eps_dec=0.00035))
    agents[i].load_model('./model_dumps/2021-01-02-21:06:39Papa-0.0001.torch')

cars = []
for i in range(0, NB_AGENTS):
    cars.append(Car(init_pos=map.player, map=map))

print(str(datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))


if PLAY:
    n_games = 1

obs_s_ = [None for i in range(NB_AGENTS)]
reward_ = [None for i in range(NB_AGENTS)]
actions_s = [None for i in range(NB_AGENTS)]


def play(n_game):
    for i in range(n_game):
        scores_ = [0 for i in range(NB_AGENTS)]
        done_ = [False for i in range(NB_AGENTS)]
        obs_s = []
        for car in cars:
            car.reset()
        if SHOW:
            SHOW_RATE = True

        for car in cars:
            car.move(2)

        for car in cars:
            obs_s.append(car.get_state())

        while not all([done for done in done_]):

            if SHOW and SHOW_RATE:
                screen.fill('grey')

            if not PLAY:
                for index_agent, agent in enumerate(agents):
                    if not done_[index_agent]:
                        actions_s[index_agent] = agent.chooseAction(obs_s[index_agent])
                        obs_, reward, done = cars[index_agent].move(actions_s[index_agent])
                        obs_s_[index_agent] = obs_
                        reward_[index_agent] = reward
                        done_[index_agent] = done

            for index_car, car in enumerate(agents):
                scores_[index_car] += reward_[index_car]
                car.score = scores_[index_car]

            if not LOAD and not PLAY:
                for index_car, car in enumerate(agents):
                    agent.storeTransition(obs_s[index_car], actions_s[index_car], reward_[index_car], obs_s_[index_car], done_[index_car])
                    agent.learn()

            if (SHOW and SHOW_RATE) or PLAY:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game_engine_status = False

                map.draw(screen, None)
                for index_car, car in enumerate(cars):
                    if not done_[index_car]:
                        car.draw(screen)
                pygame.display.update()

            for index_car, car in enumerate(cars):
                if obs_s_[index_car] is None:
                    obs_s_[index_car] = car.get_state()

                obs_s[index_car] = obs_s_[index_car]

        #scores.append(score)
        #eps_history.append(agent.epsilon)

        #avg_score = np.mean(scores[-100:])
        #avg_scores.append(avg_score)

        for index_agent, agent in enumerate(agents):
            agent.decay_epsilon()

        if LOAD:
            exit(0)


play(n_games)

end = time.time()
total = end - start
print('Took {} for {}, avg {}'.format(total, n_games, total / n_games))


gen_date = str(datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
if SAVE and not PLAY:
    torch.save(agent.Q_next.state_dict(), './model_dumps/' + gen_date + '{}-{}.torch'.format(slug, lr_arg))

    generations = [i for i in range(0, n_games)]

    fig, ax = plt.subplots()
    ax.plot(generations, scores, label="scores")
    ax.plot(generations, avg_scores, label="avg score")
    plt.savefig('./graph/{}-{}-1-{}.png'.format(slug, lr_arg, gen_date))

    fig2, ax2 = plt.subplots()
    ax2.plot(generations, eps_history, label="epsilon")

    plt.savefig('./graph/{}-{}-2-{}.png'.format(slug, lr_arg, gen_date))

exit(0)
