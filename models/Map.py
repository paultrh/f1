import pickle
import math
from config import SHOW

if SHOW:
    import pygame

    pygame.font.init()
    font = pygame.font.get_default_font()
    GAME_FONT = pygame.font.SysFont(font, 20)

X_FACT = 8
Y_FACT = 6

MAP_STATE = {
    'bahrain': {'kills': [], 'nb_finish': 0, 'last_run': False},
    'barcelona': {'kills': [], 'nb_finish': 0, 'last_run': False},
    'melbourne': {'kills': [], 'nb_finish': 0, 'last_run': False},
    'shanghai': {'kills': [], 'nb_finish': 0, 'last_run': False},
    'monaco': {'kills': [], 'nb_finish': 0, 'last_run': False},
}

class Map:

    # Build Reward gates and tracks
    def __init__(self, tile_file, map_name):

        self.player = (100, 100)
        self.kills = MAP_STATE[map_name]['kills']
        self.nb_finish = MAP_STATE[map_name]['nb_finish']

        with open(tile_file, 'rb') as filehandle:
            _map = pickle.load(filehandle)
            self.player = _map['pos']
            self.walls = _map['walls']
            self.boosts = _map['boosts']
            self.final = _map['final']
            self.init_angle = _map['init_angle']
            self.nb_lap = _map['nb_lap']

            self.name = map_name

    def update(self):
        pass

    def Pol(self, screen, Xa, Ya, Xb, Yb, W):
        Dx = Xb - Xa
        Dy = Yb - Ya
        D = math.sqrt(Dx * Dx + Dy * Dy)
        Dx = W * Dx / D
        Dy = W * Dy / D

        p1 = Xa - Dy, Ya + Dx
        p2 = Xa + Dy, Ya - Dx
        p3 = Xb - Dy, Yb + Dx
        p4 = Xb + Dy, Yb - Dx

        pygame.draw.line(screen, (255, 0, 255), p1, p2)
        pygame.draw.line(screen, (255, 0, 255), p2, p4)
        pygame.draw.line(screen, (255, 0, 255), p4, p3)
        pygame.draw.line(screen, (255, 0, 255), p3, p1)

    def draw(self, screen, car1):
        for wall in self.walls:
            # self.Pol(screen, wall[0][0], wall[0][1], wall[1][0], wall[1][1], 20)
            pygame.draw.line(screen, (0, 0, 0), wall[0], wall[1])

        for kill in self.kills:
            pygame.draw.circle(screen, (255, 0, 0), (kill[0], kill[1]), 2)

        if car1 is not None:
            for boost in self.boosts:
                boost_id = str(boost[0][0]) + str(boost[0][1])
                if boost_id not in car1.collected_boost:
                    pygame.draw.line(screen, (255, 0, 255), boost[0], boost[1])

        if len(self.final) == 2:
            pygame.draw.line(screen, (0, 0, 255), self.final[0], self.final[1])

        name = GAME_FONT.render('name: {}'.format(self.name), False, (0, 0, 0))
        nd_finish = GAME_FONT.render('nb_finish: {}'.format(self.nb_finish), False, (0, 0, 0))
        screen.blit(name, (500, 10))
        screen.blit(nd_finish, (500, 20))
