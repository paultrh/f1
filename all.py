import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
import math


SHOW = False

if SHOW:
    import pygame

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        #observation = observation.view(-1)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

class Agent(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=1000000, eps_end=0.03, eps_dec=0.0001):

        self.losses = []
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_MIN = eps_end
        self.EPS_DEC = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.Q_eval = DeepQNetwork(lr, n_actions=self.n_actions,
                              input_dims=input_dims, fc1_dims=128, fc2_dims=128)
        self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.new_state_memory = np.zeros((self.mem_size, *input_dims))
        self.action_memory = np.zeros((self.mem_size, self.n_actions),
                                      dtype=np.uint8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def storeTransition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - terminal
        self.mem_cntr += 1

    def load_model(self, path):
        self.Q_eval.load_state_dict(T.load(path))
        self.Q_eval.eval()

    def chooseAction(self, observation, no_rnd=False):
        rand = np.random.random()
        rd = False
        actions = self.Q_eval.forward(observation)
        if rand > self.EPSILON or no_rnd:
            action = T.argmax(actions).item()
        else:
            rd = True
            action = np.random.choice(self.action_space)
        return action, rd

    def decay_epsilon(self):
        self.EPSILON = self.EPSILON - (self.EPSILON * self.EPS_DEC) if self.EPSILON > \
                                                      self.EPS_MIN else self.EPS_MIN

    def learn(self):
        if self.mem_cntr > self.batch_size:
            self.Q_eval.optimizer.zero_grad()

            max_mem = self.mem_cntr if self.mem_cntr < self.mem_size \
                                    else self.mem_size

            batch = np.random.choice(max_mem, self.batch_size)
            state_batch = self.state_memory[batch]
            action_batch = self.action_memory[batch]
            action_values = np.array(self.action_space, dtype=np.int32)
            action_indices = np.dot(action_batch, action_values)
            reward_batch = self.reward_memory[batch]
            new_state_batch = self.new_state_memory[batch]
            terminal_batch = self.terminal_memory[batch]

            reward_batch = T.Tensor(reward_batch).to(self.Q_eval.device)
            terminal_batch = T.Tensor(terminal_batch).to(self.Q_eval.device)

            q_eval = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            #q_target = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            q_target = q_eval.clone()
            q_next = self.Q_eval.forward(new_state_batch).to(self.Q_eval.device)

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            q_target[batch_index, action_indices] = reward_batch + \
                                self.GAMMA*T.max(q_next, dim=1)[0]*terminal_batch

            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            self.losses.append(loss.item())
            loss.backward()
            for param in self.Q_eval.parameters():
                param.grad.data.clamp_(-1, 1)
            self.Q_eval.optimizer.step()



X_FACT = 8
Y_FACT = 6

class Map:

    # Build Reward gates and tracks
    def __init__(self, tile_file):

        self.player = (100, 100)
        self.kills = []

        with open(tile_file, 'rb') as filehandle:
            _map = pickle.load(filehandle)
            self.player = _map['pos']
            self.walls = _map['walls']
            self.boosts = _map['boosts']
            self.final = _map['final']
            self.init_angle = _map['init_angle']
            self.nb_lap = _map['nb_lap']

    def update(self):
        pass


    def draw(self, screen, car1):
        for wall in self.walls:
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


shape_width = int(463 / 50)
shape_height = int(1010 / 50)

if SHOW:
    import pygame

    pygame.font.init()
    font = pygame.font.get_default_font()
    GAME_FONT = pygame.font.SysFont(font, 20)

    car_img_raw = pygame.image.load('assets/cars/pitstop_car_1.png')
    car_img_scale = pygame.transform.scale(car_img_raw, (shape_width, shape_height))
    car_img_final = pygame.transform.flip(car_img_scale, False, True)


class Car:

    def __init__(self, init_pos, map):
        self.map = map

        self.init_pos = init_pos
        self.velocity = -0.5
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
        self.sensor_1, self.sensor_2, self.sensor_3, self.sensor_4, self.sensor_5, self.sensor_6, self.sensor_7, self.sensor_8, self.sensor_9,  = None, None, None, None, None, None, None, None, None
        self.sensors = [self.sensor_1, self.sensor_2, self.sensor_3, self.sensor_4, self.sensor_5]
        self.compute_sensors()

        self.sensor_coord_hit_wall, self.sensor_coord_hit_reward, self.sensor_coord_hit_finish, self.sensor_hit_reward, self.sensor_hit_wall, self.sensor_hit_finish  = [], [], [], [], [], []
        self.init_sensors()

        # Car boosts
        self.collected_boost = set()

    def init_sensors(self):

        self.sensor_coord_hit_wall = [(-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1)]
        self.sensor_coord_hit_finish = [(-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1)]
        self.sensor_coord_hit_reward = [(-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1)]
        self.sensor_hit_reward = [self.out, self.out, self.out, self.out, self.out, self.out, self.out, self.out, self.out]
        self.sensor_hit_wall = [self.out, self.out, self.out, self.out, self.out, self.out, self.out, self.out, self.out]
        self.sensor_hit_finish = [self.out, self.out, self.out, self.out, self.out, self.out, self.out, self.out, self.out]

    def compute_sensors(self):
        cx = self.cx
        cy = self.cy
        angle = self.angle
        sensor_size = self.sensor_size

        center_front = rotate_point(self.pos_x + shape_width / 2, self.pos_y, cx, cy, angle)
        middle_left = rotate_point(self.pos_x, self.pos_y + shape_width / 2, cx, cy, angle)
        middle_right = rotate_point(self.pos_x + shape_width, self.pos_y + shape_width / 2, cx, cy, angle)

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

        self.sensors = np.array([self.sensor_1, self.sensor_2, self.sensor_3, self.sensor_4, self.sensor_5, self.sensor_6, self.sensor_7, self.sensor_8, self.sensor_9])
        #self.sensors = np.array(
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
        return np.concatenate((self.sensor_hit_wall, [self.velocity, self.fuel]), axis=0)



    def draw(self, screen):

        #pygame.draw.circle(screen, (255, 255, 0), (self.cx, self.cy), 7)

        for sensor in self.sensors:
            pygame.draw.line(screen, (0, 255, 0), sensor[0], sensor[1])

        for hit in self.sensor_coord_hit_wall:
            pygame.draw.circle(screen, (255, 0, 0), (hit[0], hit[1]), 2)

        #for box in self.box:
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
            self.velocity += 0.1

        if self.velocity < -1.6:
            self.velocity = -1.6
        elif self.velocity > -0.2:
            self.velocity = -0.2

        self.update()
        reward = self.process_car_status(it)

        if reward == 10000 + self.fuel:
            done = True

        if self.collide:
            done = True

        return self.get_state(), reward, done

    # Function update car status and compute reward based on new position
    def process_car_status(self, it):
        reward = +10 + 10 * -self.velocity

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
                self.sensor_hit_wall[idx_sensor] = sensor_collide
                self.sensor_coord_hit_wall[idx_sensor] = sensor_collide_coord

                #if 20 > sensor_collide:
                #    reward = -(100 - (sensor_collide * 2))

                if 1 > sensor_collide > -150:
                    self.collide = True
                    reward = -10000


        # Sensor detect rewards
        for idx_sensor, sensor in enumerate(self.sensors):
            sensor_collide = None
            sensor_collide_coord = None
            old_dist = None
            cloosest_boost_id = None
            for boost in self.map.boosts:
                boost_id = str(boost[0][0]) + str(boost[0][1])
                if boost_id in self.collected_boost:
                    continue
                temp_collide = getCollisionPoint(boost[0][0], boost[0][1], boost[1][0], boost[1][1],
                                                 sensor[0][0], sensor[0][1], sensor[1][0], sensor[1][1])
                if temp_collide is not None:
                    new_dist = dist(sensor[0][0], sensor[0][1], temp_collide[0], temp_collide[1])
                    if old_dist is None or (
                            old_dist is not None and new_dist < old_dist):
                        sensor_collide = new_dist
                        sensor_collide_coord = temp_collide
                        old_dist = sensor_collide
                        cloosest_boost_id = boost_id
            if sensor_collide is not None:
                self.sensor_hit_reward[idx_sensor] = sensor_collide
                self.sensor_coord_hit_reward[idx_sensor] = sensor_collide_coord

                if 1 > sensor_collide > -150:
                    self.collected_boost.add(cloosest_boost_id)
                    reward = +5000

        # Sensor detect rewards
        for idx_sensor, sensor in enumerate(self.sensors):
            sensor_collide = None
            sensor_collide_coord = None
            temp_collide = getCollisionPoint(self.map.final[0][0], self.map.final[0][1], self.map.final[1][0], self.map.final[1][1],
                                             sensor[0][0], sensor[0][1], sensor[1][0], sensor[1][1])
            if temp_collide is not None:
                new_dist = dist(sensor[0][0], sensor[0][1], temp_collide[0], temp_collide[1])
                sensor_collide = new_dist
                sensor_collide_coord = temp_collide
            if sensor_collide is not None:
                self.sensor_hit_finish[idx_sensor] = sensor_collide
                self.sensor_coord_hit_finish[idx_sensor] = sensor_collide_coord

                if 1 > sensor_collide > -150:
                    reward = +10000 + self.fuel

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



def linesCollided(x1, y1, x2, y2, x3, y3, x4, y4):
    try:
        uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        if 0 <= uA <= 1 and 0 <= uB <= 1:
            return True
        return False
    except Exception:
        return False



def getCollisionPoint(x1, y1, x2, y2, x3, y3, x4, y4):
    try:
        uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        if 0 <= uA <= 1 and 0 <= uB <= 1:
            intersectionX = x1 + (uA * (x2 - x1))
            intersectionY = y1 + (uA * (y2 - y1))
            return intersectionX, intersectionY
        return None
    except Exception:
        return None


def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def normalise(data, sensor_size):
    return data / sensor_size


def rotate_point(old_x, old_y, cx, cy, angle):
    tempX = old_x - cx
    tempY = old_y - cy

    # now apply rotation
    rotatedX = tempX * math.cos(math.radians(-angle)) - tempY * math.sin(math.radians(-angle))
    rotatedY = tempX * math.sin(math.radians(-angle)) + tempY * math.cos(math.radians(-angle))

    x = rotatedX + cx
    y = rotatedY + cy

    return x, y

import time
from datetime import datetime
import random

LOAD = False
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

slug = random.choice(slugs)
lr_arg = 0.0001


start = time.time()

map = Map('loops.map')
car1 = Car(init_pos=map.player, map=map)
agent = Agent(gamma=0.99, epsilon=0.4, lr=lr_arg, input_dims=[11], batch_size=1024, n_actions=5, eps_dec=0.006)


scores, avg_scores, eps_history, its = [], [], [], []
n_games = 100

print("starting with lr {}".format(lr_arg))

if PLAY:
    n_games = 1

def play(n_game):
    for i in range(n_game):
        score = 0
        it = 0
        done = False
        car1.reset()
        if SHOW:
            SHOW_RATE = i % 100 == 0
        car1.move(3, 0)
        obs = car1.get_state()
        while not done:
            it += 1

            if it > 5000:
                SHOW_RATE = True

            if SHOW and SHOW_RATE:
                screen.fill('grey')

            action, rd = agent.chooseAction(obs)
            obs_, reward, done = car1.move(action, it)

            score += reward
            car1.score = score

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

        map.kills.append((car1.pos_x, car1.pos_y))

        agent.decay_epsilon()

        scores.append(score)
        eps_history.append(agent.EPSILON)

        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        #agent.decrement_linear_epsilon()
        #agent.decrement_expo_epsilon()

        print('episode {}; it: {}, score: {}, avg score {}, epsilon {}'.format(i, it, score, avg_score, agent.EPSILON))


play(n_games)

end = time.time()
total = end - start
print('Took {} for {}, avg {}'.format(total, n_games, total / n_games))

gen_date = str(datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
torch.save(agent.Q_eval.state_dict(), './model_dumps/' + gen_date + '{}-{}.torch'.format(slug, lr_arg))
