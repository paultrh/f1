import numpy as np
import math
from tools.physics import linesCollided, getCollisionPoint, dist, rotate_point, normalise

from config import SHOW

shape_width = int(463 / 50)
shape_height = int(1010 / 50)

if SHOW:
    import pygame

    pygame.font.init()
    font = pygame.font.get_default_font()
    GAME_FONT = pygame.font.SysFont(font, 20)

    car_sprite = pygame.sprite.Group()

    car_img_raw = pygame.image.load('assets/cars/pitstop_car_1.png')
    car_img_scale = pygame.transform.scale(car_img_raw, (shape_width, shape_height))
    car_img_final = pygame.transform.flip(car_img_scale, False, True)

REWARD_WIN = +10000
REWARD_LOSE = -10000


# create Explosion class
class Explosion(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.images = []
        for num in range(1, 6):
            img = pygame.image.load(f"explosion/img/exp{num}.png")
            img = pygame.transform.scale(img, (100, 100))
            self.images.append(img)
        self.index = 0
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]
        self.counter = 0

    def update(self):
        explosion_speed = 4
        # update explosion animation
        self.counter += 1

        if self.counter >= explosion_speed and self.index < len(self.images) - 1:
            self.counter = 0
            self.index += 1
            self.image = self.images[self.index]

        # if the animation is complete, reset animation index
        if self.index >= len(self.images) - 1 and self.counter >= explosion_speed:
            self.kill()


class Car:

    def __init__(self, init_pos, map):
        self.map = map

        self.init_pos = init_pos
        self.velocity = -0.8
        self.pos_x = init_pos[0]
        self.pos_y = init_pos[1]
        self.cx = self.pos_x + shape_width / 2
        self.cy = self.pos_y + shape_width / 2
        self.angle = self.map.init_angle
        self.nb_lap = self.map.nb_lap
        self.speed = 0
        self.score = 0
        self.sensor_size = 1500
        self.out = self.sensor_size
        self.collide = False
        self.fuel = 5000
        self.latest_reward = 0

        self.last_100_pos = []

        # Car collidingBox
        self.p1, self.p2, self.p3, self.p4 = None, None, None, None
        self.p5, self.p6, self.p7, self.p8, self.p9, self.p10, self.p11, self.p12 = None, None, None, None, None, None, None, None
        self.box = None
        self.compute_bounding_box()

        # Car sensors
        self.sensor_1, self.sensor_2, self.sensor_3, self.sensor_4, self.sensor_5, self.sensor_6, self.sensor_7, self.sensor_8, self.sensor_9, = None, None, None, None, None, None, None, None, None
        self.sensors = [self.sensor_1, self.sensor_2, self.sensor_3, self.sensor_4, self.sensor_5]
        self.compute_sensors()

        self.sensor_coord_hit_wall, self.sensor_coord_hit_reward, self.sensor_coord_hit_finish, self.sensor_hit_reward, self.sensor_hit_wall, self.sensor_hit_finish = [], [], [], [], [], []
        self.init_sensors()

        # Car boosts
        self.collected_boost = set()

    def init_sensors(self):

        self.sensor_coord_hit_wall = [(-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
                                      (-1, -1)]
        self.sensor_coord_hit_finish = [(-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
                                        (-1, -1)]
        self.sensor_coord_hit_reward = [(-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
                                        (-1, -1)]
        self.sensor_hit_reward = [self.out, self.out, self.out, self.out, self.out, self.out, self.out, self.out,
                                  self.out]
        self.sensor_hit_wall = [self.out, self.out, self.out, self.out, self.out, self.out, self.out, self.out,
                                self.out]
        self.sensor_hit_finish = [self.out, self.out, self.out, self.out, self.out, self.out, self.out, self.out,
                                  self.out]

    def compute_sensors(self):
        cx = self.cx
        cy = self.cy
        angle = self.angle
        sensor_size = self.sensor_size

        center_front = rotate_point(self.pos_x + shape_width / 2, self.pos_y - shape_height / 2, cx, cy, angle)
        middle_left = rotate_point(self.pos_x + shape_width / 2, self.pos_y + shape_width / 2, cx, cy, angle)
        middle_right = rotate_point(self.pos_x + shape_width / 2, self.pos_y + shape_width / 2, cx, cy, angle)

        # Forward
        self.sensor_1 = (center_front, rotate_point(cx, cy - sensor_size, cx, cy, angle))

        # Lateral forward
        self.sensor_2 = (middle_left, rotate_point(cx - sensor_size / 2, cy - sensor_size, cx, cy, angle))
        self.sensor_3 = (middle_right, rotate_point(cx + sensor_size / 2, cy - sensor_size, cx, cy, angle))

        # Lateral medium
        self.sensor_4 = (middle_left, rotate_point(cx - sensor_size, cy - sensor_size, cx, cy, angle))
        self.sensor_5 = (middle_right, rotate_point(cx + sensor_size, cy - sensor_size, cx, cy, angle))

        # Lateral low
        self.sensor_6 = (middle_left, rotate_point(cx - sensor_size * 1.5, cy - sensor_size, cx, cy, angle))
        self.sensor_7 = (middle_right, rotate_point(cx + sensor_size * 1.5, cy - sensor_size, cx, cy, angle))

        # Lateral baseline
        self.sensor_8 = (middle_left, rotate_point(cx - sensor_size, cy, cx, cy, angle))
        self.sensor_9 = (middle_right, rotate_point(cx + sensor_size, cy, cx, cy, angle))

        self.sensors = np.array(
            [self.sensor_1, self.sensor_2, self.sensor_3, self.sensor_4, self.sensor_5, self.sensor_6, self.sensor_7,
             self.sensor_8, self.sensor_9])
        # self.sensors = np.array(
        #    [self.sensor_1, self.sensor_8, self.sensor_9])

    def compute_bounding_box(self):
        cx = self.cx
        cy = self.cy
        pos_x = self.pos_x
        pos_y = self.pos_y
        angle = self.angle

        self.p1 = rotate_point(pos_x, pos_y, cx, cy, angle)
        self.p2 = rotate_point(pos_x + shape_width, pos_y, cx, cy, angle)

        self.p3 = rotate_point(pos_x + shape_width, pos_y, cx, cy, angle)
        self.p4 = rotate_point(pos_x + shape_width, pos_y + shape_width, cx, cy, angle)

        self.p5 = rotate_point(pos_x + shape_width, pos_y + shape_width, cx, cy, angle)
        self.p6 = rotate_point(pos_x, pos_y + shape_width, cx, cy, angle)

        self.p7 = rotate_point(pos_x, pos_y + shape_width, cx, cy, angle)
        self.p8 = rotate_point(pos_x, pos_y, cx, cy, angle)

        self.box = np.array([(self.p1, self.p2), (self.p3, self.p4), (self.p5, self.p6), (self.p7, self.p8)])

    def get_state(self):
        return np.concatenate((self.sensor_hit_wall, self.sensor_hit_finish, [self.velocity]), axis=0)
        # return np.concatenate((self.sensor_hit_wall, self.sensor_hit_reward, self.sensor_hit_finish, [self.velocity]), axis=0)
        # return self.sensor_hit_wall

    def draw(self, screen):
        car_sprite.draw(screen)
        car_sprite.update()

        # pygame.draw.circle(screen, (255, 255, 0), (self.cx, self.cy), 7)
        if self.collide:
            #pygame.mixer.music.load('sound/crash.mp3')
            #pygame.mixer.music.play(0)
            car_sprite.add(Explosion(self.pos_x, self.pos_y))

        for sensor in self.sensors:
            pygame.draw.line(screen, (0, 255, 0), sensor[0], sensor[1])

        for hit in self.sensor_coord_hit_wall:
            pygame.draw.circle(screen, (255, 0, 0), (hit[0], hit[1]), 2)

        # for box in self.box:
        #    pygame.draw.line(screen, (0, 0, 255), box[0], box[1])

        for hit in self.sensor_coord_hit_reward:
            pygame.draw.circle(screen, (0, 0, 255), (hit[0], hit[1]), 2)

        for hit in self.sensor_coord_hit_finish:
            pygame.draw.circle(screen, (133, 133, 133), (hit[0], hit[1]), 2)

        rotated_image = pygame.transform.rotate(car_img_final, self.angle)
        rot_point = rotate_point(self.cx, self.cy, self.cx, self.cy, self.angle)
        new_rect = rotated_image.get_rect(center=car_img_final.get_rect(center=(rot_point[0], rot_point[1])).center)
        screen.blit(rotated_image, new_rect.topleft)

        text_x = GAME_FONT.render('x: {}'.format(self.pos_x), False, (0, 0, 0))
        text_y = GAME_FONT.render('y: {}'.format(self.pos_y), False, (0, 0, 0))
        text_angle = GAME_FONT.render('angle: {}'.format(self.angle), False, (0, 0, 0))
        velocity = GAME_FONT.render('velocity: {}'.format(self.velocity), False, (0, 0, 0))
        fuel = GAME_FONT.render('fuel: {}'.format(self.fuel), False, (0, 0, 0))
        score = GAME_FONT.render('score: {}'.format(self.score), False, (0, 0, 0))

        curr_reward = GAME_FONT.render('reward: {}'.format(self.latest_reward), False, (0, 0, 0))

        screen.blit(text_x, (10, 10))
        screen.blit(text_y, (10, 20))
        screen.blit(text_angle, (10, 30))
        screen.blit(velocity, (10, 40))
        screen.blit(fuel, (10, 50))
        screen.blit(score, (10, 60))
        screen.blit(curr_reward, (10, 70))

        fx = self.get_state()
        sensor_wall_1 = GAME_FONT.render(str(int(fx[0])), False, (0, 0, 0))
        sensor_wall_2 = GAME_FONT.render(str(int(fx[1])), False, (0, 0, 0))
        sensor_wall_3 = GAME_FONT.render(str(int(fx[2])), False, (0, 0, 0))
        sensor_wall_4 = GAME_FONT.render(str(int(fx[3])), False, (0, 0, 0))
        sensor_wall_5 = GAME_FONT.render(str(int(fx[4])), False, (0, 0, 0))
        sensor_wall_6 = GAME_FONT.render(str(int(fx[5])), False, (0, 0, 0))
        sensor_wall_7 = GAME_FONT.render(str(int(fx[6])), False, (0, 0, 0))
        sensor_wall_8 = GAME_FONT.render(str(int(fx[7])), False, (0, 0, 0))
        sensor_wall_9 = GAME_FONT.render(str(int(fx[8])), False, (0, 0, 0))

        screen.blit(sensor_wall_1, (300, 450))

        screen.blit(sensor_wall_2, (275, 500))
        screen.blit(sensor_wall_3, (325, 500))

        screen.blit(sensor_wall_4, (250, 550))
        screen.blit(sensor_wall_5, (350, 550))

        screen.blit(sensor_wall_6, (250, 600))
        screen.blit(sensor_wall_7, (350, 600))

        screen.blit(sensor_wall_8, (250, 650))
        screen.blit(sensor_wall_9, (350, 650))


    # Function update car status and compute reward based on new position
    def move(self, action, it):
        done = False
        self.fuel = self.fuel - 1

        if action == 0:
            # LEFT
            self.angle += 2
        if action == 1:
            # RIGHT
            self.angle -= 2
        if action == 2:
            # FAST
            self.velocity -= 0.1
        if action == 3:
            # SLOW
            self.velocity += 0.3

        if self.velocity < -2:
            self.velocity = -2
        elif self.velocity > -0.2:
            self.velocity = -0.2

        self.update()
        reward = self.process_car_status(it)

        if reward == REWARD_WIN:
            done = True

        if self.collide or self.fuel == 0:
            done = True

        return self.get_state(), reward, done

    # Function update car status and compute reward based on new position
    def process_car_status(self, it):
        reward = 100 * -self.velocity

        # Sensor detect walls
        for idx_sensor, sensor in enumerate(self.sensors):
            sensor_collide_coord = None
            sensor_collide = None
            old_dist = None
            for wall in self.map.walls:
                temp_collide = getCollisionPoint(wall[0][0], wall[0][1], wall[1][0], wall[1][1],
                                                 sensor[0][0], sensor[0][1], sensor[1][0], sensor[1][1])
                if temp_collide is not None:
                    new_dist = dist(sensor[0][0], sensor[0][1], temp_collide[0], temp_collide[1])
                    if old_dist is None or (
                            old_dist is not None and new_dist < old_dist):
                        sensor_collide = new_dist
                        sensor_collide_coord = temp_collide
                        old_dist = sensor_collide
            if sensor_collide is not None:
                self.sensor_hit_wall[idx_sensor] = max(sensor_collide - 4, 0)
                self.sensor_coord_hit_wall[idx_sensor] = sensor_collide_coord

                # if 20 > sensor_collide:
                #    reward = -(100 - (sensor_collide * 2))

                if 4 > sensor_collide > -150:
                    self.collide = True
                    reward = REWARD_LOSE

        # Sensor detect rewards
        # for idx_sensor, sensor in enumerate(self.sensors):
        #    sensor_collide = None
        #    sensor_collide_coord = None
        #    old_dist = None
        #    cloosest_boost_id = None
        #    for boost in self.map.boosts:
        #        boost_id = str(boost[0][0]) + str(boost[0][1])
        #        if boost_id in self.collected_boost:
        #            continue
        #        temp_collide = getCollisionPoint(boost[0][0], boost[0][1], boost[1][0], boost[1][1],
        #                                         sensor[0][0], sensor[0][1], sensor[1][0], sensor[1][1])
        #        if temp_collide is not None:
        #            new_dist = dist(sensor[0][0], sensor[0][1], temp_collide[0], temp_collide[1])
        #            if old_dist is None or (
        #                    old_dist is not None and new_dist < old_dist):
        #                sensor_collide = new_dist
        #                sensor_collide_coord = temp_collide
        #                old_dist = sensor_collide
        #                cloosest_boost_id = boost_id
        #    if sensor_collide is not None:
        #        self.sensor_hit_reward[idx_sensor] = sensor_collide
        #        self.sensor_coord_hit_reward[idx_sensor] = sensor_collide_coord

        #        if 1 > sensor_collide > -150:
        #            self.collected_boost.add(cloosest_boost_id)
        #            reward = +5000

        # Sensor detect rewards
        for idx_sensor, sensor in enumerate(self.sensors):
            sensor_collide = None
            sensor_collide_coord = None
            temp_collide = getCollisionPoint(self.map.final[0][0], self.map.final[0][1], self.map.final[1][0],
                                             self.map.final[1][1],
                                             sensor[0][0], sensor[0][1], sensor[1][0], sensor[1][1])
            if temp_collide is not None:
                new_dist = dist(sensor[0][0], sensor[0][1], temp_collide[0], temp_collide[1])
                sensor_collide = new_dist
                sensor_collide_coord = temp_collide
            if sensor_collide is not None and self.sensor_hit_wall[idx_sensor] > sensor_collide:
                self.sensor_hit_finish[idx_sensor] = max(sensor_collide - 4, 0)
                self.sensor_coord_hit_finish[idx_sensor] = sensor_collide_coord

                if 4 > sensor_collide > -150:
                    reward = REWARD_WIN

        self.latest_reward = reward
        return reward

    def update(self):
        self.init_sensors()

        old_x, old_y = self.pos_x, self.pos_y
        angle = self.angle

        delta_y = self.velocity * math.cos(math.radians(angle))
        delta_x = self.velocity * math.sin(math.radians(angle))

        new_x = old_x + delta_x
        new_y = old_y + delta_y

        self.pos_x = new_x
        self.pos_y = new_y

        self.cx = self.pos_x + shape_width / 2
        self.cy = self.pos_y + shape_width / 2

        self.compute_bounding_box()
        self.compute_sensors()

    def reset(self):
        self.__init__(self.init_pos, self.map)
