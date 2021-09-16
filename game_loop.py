import random

from ai.ddqn import Agent
from ai.v2ddqn import AgentV2
from config import SHOW
import appnope
appnope.nope()
import time
from datetime import datetime
from models.Car import Car, REWARD_WIN
from models.Map import Map, MAP_STATE

import torch

import matplotlib.pyplot as plt
import numpy as np
import sys


LOAD = False
RND = True
SAVE = True
PLAY = False
REPLAY = None

if SHOW or PLAY:
    import pygame
    pygame.init()
    pygame.display.set_caption("F1")

    screen = pygame.display.set_mode((920, 720))


init_pos = (110, 200)


slugs = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot', 'Golf', 'Hotel', 'India', 'Juliet', 'Kilo', 'Lima',
         'Mike', 'November', 'Oscar', 'Papa', 'Quebec', 'Romeo', 'Sierra', 'Tango', 'Uniform', 'Victor', 'Whiskey',
         'X-ray', 'Yankee', 'Zulu']

game_engine_status = True

if len(sys.argv) == 3:
    slug = sys.argv[1]
    lr_arg = float(sys.argv[2])
else:
    slug = random.choice(slugs)
    lr_arg = 0.0001

start = time.time()

agent = AgentV2(gamma=0.99, epsilon=0.5, lr=lr_arg, input_dims=[19], batch_size=4096, n_actions=4, eps_dec=0.006)

if LOAD: # romeo
    agent.load_model('./model_dumps/2021-01-06-10:30:09Delta-0.0001.torch')

scores, avg_scores, eps_history, its = [], [], [], []
n_games = 1000

print("starting with lr {}".format(lr_arg))


if PLAY:
    n_games = 100

def play(n_game):
    for i in range(n_game):
        score = 0
        it = 0
        done = False
        won = False
        reward = 0
        last_is_rd = False
        map_name = random.choice(list(MAP_STATE.keys()))

        map = Map('maps/' + map_name + '.map', map_name)
        car1 = Car(init_pos=map.player, map=map)

        car1.reset()
        if SHOW:
            SHOW_RATE = i % 1 == 0
        car1.move(3, 0)
        obs = car1.get_state()
        while not done:
            it += 1

            if it > 5000:
                SHOW_RATE = True

            if SHOW and SHOW_RATE:
                screen.fill('grey')

            action, rd = agent.chooseAction(obs, no_rnd=not RND)
            last_is_rd = rd
            obs_, reward, done = car1.move(action, it)

            score += reward
            car1.score = score

            if not PLAY:
                agent.storeTransition(obs, action, reward, obs_, done)
                agent.learn()

            if (SHOW and SHOW_RATE) or PLAY:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game_engine_status = False

                map.draw(screen, car1)
                car1.draw(screen)
                pygame.display.update()

            obs = obs_

        MAP_STATE[map_name]['kills'].append((car1.pos_x, car1.pos_y))
        if reward == REWARD_WIN:
            MAP_STATE[map_name]['nb_finish'] += 1
            won = True

        if last_is_rd and not won:
            print('Died of random', end=' | ')

        agent.decay_epsilon()

        scores.append(score)
        eps_history.append(agent.EPSILON)

        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        #agent.decrement_linear_epsilon()
        #agent.decrement_expo_epsilon()

        print('episode {}; it: {}, score: {}, avg score {}, epsilon {}, map {}, won {}'.format(i, it, score, avg_score, agent.EPSILON, map_name, won))


play(n_games)

end = time.time()
total = end - start
print('Took {} for {}, avg {}'.format(total, n_games, total / n_games))

for elt in MAP_STATE.keys():
    print('{} : {}'.format(elt, MAP_STATE[elt]["nb_finish"]))

gen_date = str(datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
if SAVE and not PLAY:
    torch.save(agent.Q_next.state_dict(), './model_dumps/' + gen_date + '{}-{}.torch'.format(slug, lr_arg))

    generations = [i for i in range(0, n_games)]

    fig = plt.figure()
    fig.suptitle('{} {}'.format(slug, lr_arg), fontsize=20)

    ax1 = fig.add_subplot(311)
    ax1.scatter(generations, scores, label="scores", s=2)
    ax1.plot(generations, avg_scores, label="avg score", color='red')
    #ax1.set_yscale('log')

    ax2 = fig.add_subplot(312)
    ax2.plot(generations, eps_history, label="epsilon")

    ax3 = fig.add_subplot(313)
    ax3.plot([i for i in range(0, len(agent.losses))], agent.losses, label="loss")
    ax3.set_yscale('log')

    plt.savefig('./graph/{}-{}-{}.png'.format(slug, lr_arg, gen_date))
    plt.show()

exit(0)
