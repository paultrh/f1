import pygame
from models.Car import Car
from ai.ddqn import Agent
from models.Map import Map

LOAD = './model_dumps/2020-12-15-20:56:33November-0.001.torch'

pygame.init()
pygame.display.set_caption("F1")
screen = pygame.display.set_mode((920, 720))

map = Map('maps/test_circle.map')
car1 = Car(init_pos=map.player, map=map)

lr_arg = 0.001
agent = Agent(gamma=0.999, epsilon=0.5, lr=lr_arg, input_dims=[3], batch_size=1024, n_actions=3, eps_dec=0.006)
agent.load_model(LOAD)



for degree in [0, 90]:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_engine_status = False

    for i in range(50, 750):
        print(i)
        for j in range(100, 500):
            car = Car(init_pos=(i, j), map=map)
            car.angle = degree
            car.move(3)
            action, rd = agent.chooseAction(car.get_state(), True)
            if action == 1:
                # RED / GAUCHE
                pygame.draw.line(screen, (255, 0, 0), (i, j), (i, j))
            elif action == 2:
                # VERT / DROITE
                pygame.draw.line(screen, (0, 255, 0), (i, j), (i, j))
            else:
                # BLEU / RIEN
                pygame.draw.line(screen, (0, 0, 255), (i, j), (i, j))

        pygame.display.update()
    map.draw(screen, Car(init_pos=(20, 20), map=map))
    pygame.display.update()

    pygame.image.save(screen, "./graph/screen-{}.jpeg".format(degree))
