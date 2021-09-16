import random

from ai.ddqn import Agent
from config import SHOW
import appnope
appnope.nope()
import time
from datetime import datetime
from models.Car import Car
from models.Map import Map

import torch

import matplotlib.pyplot as plt
import numpy as np
import sys


LOAD = False
SAVE = True
PLAY = False
REPLAY = None


import pygame
pygame.init()
pygame.display.set_caption("F1")

screen = pygame.display.set_mode((920, 720))

game_engine_status = True

if len(sys.argv) == 3:
    slug = sys.argv[1]
    lr_arg = float(sys.argv[2])
else:
    slug = random.choice('show')
    lr_arg = 0.001

LOAD = './model_dumps/2020-12-15-16:23:46Tango-0.001.torch'
start = time.time()

agent = Agent(gamma=0.9, epsilon=0.05, lr=lr_arg, input_dims=[3], batch_size=2048, n_actions=3, eps_dec=0.006)
agent.load_model(LOAD)

scores, avg_scores, eps_history = [], [], []
n_games = 100

print("starting with lr {}".format(lr_arg))

if PLAY:
    n_games = 1

def play(n_game):
    for i in range(n_game):
        score = 0
        it = 0
        done = False
        map = Map(random.choice(['maps/light_turns.map', 'maps/test_circle.map']))
        car1 = Car(init_pos=map.player, map=map)
        car1.reset()
        car1.move(3)
        obs = car1.get_state()
        while not done:
            it += 1

            screen.fill('grey')

            action, rd = agent.chooseAction(obs, no_rnd=True)
            obs_, reward, done = car1.move(action)


            score += reward
            car1.score = score

            agent.storeTransition(obs, action, reward, obs_, done)
            #agent.learn()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_engine_status = False

            map.draw(screen, car1)
            car1.draw(screen)
            pygame.display.update()

            obs = obs_

        agent.decay_epsilon()

        scores.append(score)
        eps_history.append(agent.EPSILON)

        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)


        print('episode {}; it: {}, score: {}, avg score {}, epsilon {}'.format(i, it, score, avg_score, agent.EPSILON))


play(n_games)
