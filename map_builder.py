import pygame
import pickle
import pygame
import math

pygame.init()
clock = pygame.time.Clock()


# up : 0
# left : 90
# bot : 180
# right : 270

font = pygame.font.get_default_font()
GAME_FONT = pygame.font.SysFont(font, 20)

Screen_Width = 800
Screen_Height = 600

screen = pygame.display.set_mode((Screen_Width, Screen_Height))

X_FACT = 8
Y_FACT = 6

class Map:

    # Build Reward gates and tracks
    def __init__(self, name):
        self.name = name

        self.walls = []
        self.boosts = []
        self.final = ()
        self.pos = (100, 100)
        self.init_angle = 0
        self.nb_lap = 1

    def Pol(self, screen, Xa, Ya, Xb, Yb, W):
        Dx = Xb - Xa
        Dy = Yb - Ya
        D = math.sqrt(Dx * Dx + Dy * Dy)
        Dx = W * Dx / D
        Dy = W * Dy / D

        Xmin = min(Xa, Xb) - abs(Dy)
        Xmax = max(Xa, Xb) + abs(Dy)
        Ymin = min(Ya, Yb) - abs(Dx)
        Ymax = max(Ya, Yb) + abs(Dx)

        #pygame.draw.circle(screen, (255, 0, 0), (Dx, Dy), 3)

        p1 = Xa - Dy, Ya + Dx
        p2 = Xa + Dy, Ya - Dx
        p3 = Xb - Dy, Yb + Dx
        p4 = Xb + Dy, Yb - Dx

        pygame.draw.line(screen, (255, 0, 255), p1, p2)
        pygame.draw.line(screen, (255, 0, 255), p2, p4)
        pygame.draw.line(screen, (255, 0, 255), p4, p3)
        pygame.draw.line(screen, (255, 0, 255), p3, p1)

    def draw(self, screen, buffer, mode, flag):
        for wall in self.walls:
            self.Pol(screen, wall[0][0], wall[0][1], wall[1][0], wall[1][1], 15)
            pygame.draw.line(screen, (0, 0, 255), wall[0], wall[1])

        for boost in self.boosts:
            pygame.draw.line(screen, (255, 0, 255), boost[0], boost[1])

        if len(self.final) == 2:
            pygame.draw.line(screen, (233, 100, 133), self.final[0], self.final[1])

        pygame.draw.circle(screen, (255, 0, 255), self.pos, 3)

        info = GAME_FONT.render('change mode with Wall/Bonus/Player/Final'.format(buffer), False, (0, 0, 0))
        screen.blit(info, (10, 10))

        buffer = GAME_FONT.render('buffer: {}'.format(buffer), False, (0, 0, 0))
        screen.blit(buffer, (10, 30))

        buffer = GAME_FONT.render('mode: {}'.format(mode), False, (0, 0, 0))
        screen.blit(buffer, (10, 50))

        buffer = GAME_FONT.render('flag: {}'.format(flag), False, (0, 0, 0))
        screen.blit(buffer, (10, 60))

        save_info = GAME_FONT.render('press s to save', False, (0, 0, 0))
        screen.blit(save_info, (10, 70))

    def save(self):
        with open(self.name, 'wb') as filehandle:
            _map = {'pos': self.pos, 'walls': self.walls, 'boosts': self.boosts, 'final': self.final, 'init_angle': self.init_angle, 'nb_lap': self.nb_lap}
            print(_map)
            pickle.dump(_map, filehandle)

map = Map('test.map')
tmp = None
mode = 'Wall'
flag = False
starttime = 0

running = True
while running:
    clock.tick(60)
    screen.fill('grey')


    for e in pygame.event.get():
        if e == pygame.QUIT or e.type == pygame.K_ESCAPE:
            Running = False

    Mouse_x, Mouse_y = pygame.mouse.get_pos()
    key = pygame.key.get_pressed()
    if not flag:
        starttime = pygame.time.get_ticks()
        if key[pygame.K_SPACE]:
            flag = True
            if mode == 'Player':
                map.pos = (Mouse_x, Mouse_y)
            elif tmp is None:
                tmp = (Mouse_x, Mouse_y)
            else:
                if mode == 'Wall':
                    map.walls.append([(Mouse_x, Mouse_y), tmp])
                    tmp = (Mouse_x, Mouse_y)
                elif mode == 'Boosts':
                    map.boosts.append([(Mouse_x, Mouse_y), tmp])
                    tmp = None
                elif mode == 'Final':
                    map.final = ((Mouse_x, Mouse_y), tmp)
                    tmp = None
        if key[pygame.K_w]:
            flag = True
            mode = 'Wall'
            tmp = None
        elif key[pygame.K_b]:
            flag = True
            tmp = None
            mode = 'Boosts'
        elif key[pygame.K_f]:
            flag = True
            tmp = None
            mode = 'Final'
        elif key[pygame.K_p]:
            flag = True
            tmp = None
            mode = 'Player'
        elif key[pygame.K_s]:
            flag = True
            running = False
    else:
        if pygame.time.get_ticks() - starttime >= 500:
            flag = False

    map.draw(screen, tmp, mode, flag)

    pygame.display.flip()

try:
    val = input("Enter starting angle: ")
    map.init_angle = int(val)

    val = input("Enter nb lap : ")
    map.nb_lap = int(val)

    map.save()
except Exception as e:
    pass

