import pygame as pg
import numpy as np
import time
import scipy.io
import os
from examples.choose_problem_HD import system, problem, config

LIGHT_GREY = (230, 230, 230)

class VisUtils:

    def __init__(self):
        # change this name:
        self.model_name = 'PMP'
        # self.model_name = 'Neural Network'
        # self.model_name = 'Empathetic-Empathetic'
        # self.model_name = 'Nonempathetic-Empathetic'
        self.screen_width = 10  # 50
        self.screen_height = 10  # 50
        self.coordinate_scale = 80
        self.zoom = 0.25  # 0.25 change the number to adjust the position of the road frame
        self.asset_location = 'assets/'
        self.fps = 24  # max framework

        self.car_width = problem.W1
        self.car_length = problem.L1
        self.road_length = problem.R1 / 2.
        self.coordinate = 'coordinates.png'

        load_path = 'examples/vehicle/data_train_a_a_1.mat'
        # load_path = 'examples/vehicle/data_E_a_a.mat'
        # load_path = 'examples/vehicle/data_NE_a_a.mat'
        # load_path = 'examples/vehicle/data_E_na_na.mat'
        # load_path = 'examples/vehicle/data_NE_na_na.mat'

        # load_path = 'examples/vehicle/data_waymo_E_E_a_na_bvp_suc.mat'
        # load_path = 'examples/vehicle/data_waymo_NE_E_a_na_bvp_suc.mat'
        self.train_data = scipy.io.loadmat(load_path)

        # for uncontrolled intersection case, set white_car's orientation as -90
        # for unprotected left turn, set white_car's orientation as -180
        # self.car_par = [{'sprite': 'grey_car_sized.png',
        #                  'state': self.new_data['X'][:1, :],
        #                  'policy': self.new_data['P'][:1, :],
        #                  'orientation': 0.},
        #                 {'sprite': 'white_car_sized.png',
        #                  'state': self.new_data['X'][1:, :],
        #                  'policy': self.new_data['P'][1:, :],
        #                  'orientation': -180.}
        #                 ]

        self.new_data = self.generate(self.train_data)

        self.T = self.new_data['t']

        self.car_par = [{'sprite': 'grey_car_sized.png',
                         'state': self.new_data['X'][:2, :],  # pos_x, vel_x
                         'orientation': 0.},
                        {'sprite': 'white_car_sized.png',
                         'state': self.new_data['X'][2:, :],  # pos_x, vel_x
                         'orientation': -90.}
                        ]

        img_width = int(self.car_width * self.coordinate_scale * self.zoom)
        img_height = int(self.car_length * self.coordinate_scale * self.zoom)

        "initialize pygame"
        pg.init()
        self.screen = pg.display.set_mode((self.screen_width * self.coordinate_scale,
                                           self.screen_height * self.coordinate_scale))

        self.car_image = [pg.transform.rotate(pg.transform.scale(pg.image.load(self.asset_location + self.car_par[i]['sprite']),
                                               (img_width, img_height)), - self.car_par[i]['orientation']) for i in range(len(self.car_par))]

        self.coor_image = pg.image.load(self.asset_location + self.coordinate)

        # we can change the number to adjust the position of the road frame
        self.origin = np.array([35, 35])  # 35, 35; 30, 30;

        # self.origin = np.array([0, 0])

        "Draw Axis Lines"

        self.screen.fill((255, 255, 255))
        self.draw_axes() # calling draw axis function
        pg.display.flip()
        pg.display.update()

    def blit_alpha(self, target, source, location, opacity):
        x = location[0]
        y = location[1]
        temp = pg.Surface((source.get_width(), source.get_height())).convert()
        temp.blit(target, (-x, -y))
        temp.blit(source, (0, 0))
        temp.set_alpha(opacity)
        target.blit(temp, location)

    def draw_frame(self):
        '''state[t] = [s_x, s_y, v_x, v_y]_t'''
        '''state = [state_t, state_t+1, ...]'''
        # Draw the current frame
        '''frame is counting which solution step'''

        steps = self.T.shape[0]  # 10/0.1 + 1 = 101
        # steps = self.T.shape[1]

        self.screen.fill((255, 255, 255))
        self.draw_axes()
        self.draw_dashed_line1()
        self.draw_dashed_line2()
        for k in range(steps - 1):
            # self.screen.fill((255, 255, 255))
            # self.draw_axes()
            # self.draw_dashed_line1()
            # self.draw_dashed_line2()
            # Draw Images
            n_agents = 2
            for i in range(n_agents):
                '''getting pos of agent: (x, y)'''
                pos_x_old = np.array(self.car_par[i]['state'][0][k])  # car x position
                pos_x_new = np.array(self.car_par[i]['state'][0][k + 1])  # get 0 and 1 element (not include 2) : (x, y)

                pos_y_old = np.array(self.car_par[i]['state'][1][k])  # car y position
                pos_y_new = np.array(self.car_par[i]['state'][1][k + 1])  # get 0 and 1 element (not include 2) : (x, y)
                '''smooth out the movement between each step'''
                pos_x = pos_x_old * (1 - k * 1. / steps) + pos_x_new * (k * 1. / steps)
                pos_y = pos_y_old * (1 - k * 1. / steps) + pos_y_new * (k * 1. / steps)

                if i == 0:
                    pos = (pos_x, pos_y)  # car position
                if i == 1:
                    # for uncontrolled intersection case, set pos = (pos, self.road_length)
                    # for unprotected left turn, set pos = (self.road_length + self.car_width, 70 - pos)
                    # pos = (self.road_length + self.car_width, 70 - pos)  # car position

                    pos = (pos_x, pos_y)  # car position
                '''transform pos'''
                pixel_pos_car = self.c2p(pos)
                size_car = self.car_image[i].get_size()

                # try with opacity
                self.blit_alpha(self.screen, self.car_image[i].convert_alpha(), (pixel_pos_car[0] - size_car[0] / 2, pixel_pos_car[1] - size_car[1] / 2), 11*k)
                # self.screen.blit(self.car_image[i].convert_alpha(),
                #                  (pixel_pos_car[0] - size_car[0] / 2, pixel_pos_car[1] - size_car[1] / 2))
                time.sleep(0.05)
                # if self.sim.decision_type == "baseline":
                #     time.sleep(0.05)
            # Annotations
            font = pg.font.SysFont("Arial", 25)
            screen_w, screen_h = self.screen.get_size()
            label_x = screen_w - 700
            label_y = 50
            label = font.render("Model Type: {}".format(self.model_name), 1, (0, 0, 0))
            self.screen.blit(label, (label_x, label_y))


            # font = pg.font.SysFont("Arial", 25, bold=True)
            # screen_w, screen_h = self.screen.get_size()
            # label_x = screen_w - 550
            # label_y = 250
            # label = font.render("Model Type: {}".format(self.model_name), 1, (0, 0, 0))
            # # self.screen.blit(label, (label_x, label_y))
            # label_y += 30
            # label = font.render("True Policy for Vehicles: {}".format('a-na'), 1, (0, 0, 0))
            # # self.screen.blit(label, (label_x, label_y))
            #
            # label_y += 30
            # label_policy_1 = font.render("Car 1 Belief: {:.2f}".format(self.car_par[0]['policy'][0][k+1]), 1, (0, 0, 0))
            # # self.screen.blit(label_policy_1, (label_x, label_y))
            # label_y += 30
            # label_policy_2 = font.render("Car 2 Belief: {:.2f}".format(self.car_par[1]['policy'][0][k+1]), 1, (0, 0, 0))
            # # self.screen.blit(label_policy_2, (label_x, label_y))

            # label_y_offset = 30
            # pos_h, speed_h = self.car_par[0]['state'][0][k+1], self.car_par[0]['state'][1][k+1]  #y axis
            # label = font.render("Car 1 position and speed: (%5.4f , %5.4f)" % (pos_h, speed_h), 1,
            #                     (0, 0, 0))
            # self.screen.blit(label, (label_x, label_y))
            # pos_m, speed_m = self.car_par[1]['state'][0][k+1], self.car_par[1]['state'][1][k+1] #x axis
            # label = font.render("Car 2 position and speed: (%5.4f , %5.4f)" % (pos_m, speed_m), 1,
            #                     (0, 0, 0))
            # self.screen.blit(label, (label_x, label_y + label_y_offset))
            #
            # label = font.render("Frame: %i" % steps, 1, (0, 0, 0))
            # self.screen.blit(label, (10, 10))

            # self.coordinate_image = self.screen.blit(self.coor_image, (screen_w - 800, screen_h - 50))

            # recording_path = 'image_recording/'
            # pg.image.save(self.screen, "%simg%03d.png" % (recording_path, k))

            "drawing the map of state distribution"
            # pg.draw.circle(self.screen, (255, 255, 255), self.c2p(self.origin), 10)  # surface,  color, (x, y),radius>=1

            # time.sleep(1)

            pg.display.flip()
            pg.display.update()

        recording_path = 'image_recording/'
        pg.image.save(self.screen, "%simg%03d.png" % (recording_path, k))

    def draw_axes(self):
        # draw lanes based on environment
        pg.draw.line(self.screen, LIGHT_GREY, self.c2p((35 + self.car_width, -50)),
                     self.c2p((35 + self.car_width, 100)), self.car_image[0].get_size()[0] * 4)


        # pg.draw.line(self.screen, LIGHT_GREY, self.c2p((35 + self.car_width / 2, -50)),
        #              self.c2p((35 + self.car_width / 2, 100)), self.car_image[0].get_size()[0] * 2)

        # for uncontrolled intersection case, set self.car_image[1].get_size()[1]
        # for unprotected left turn, set self.car_image[1].get_size()[0]

        pg.draw.line(self.screen, LIGHT_GREY, self.c2p((100, 35 - self.car_width)),
                     self.c2p((-50, 35 - self.car_width)), self.car_image[1].get_size()[1] * 4)


        # pg.draw.line(self.screen, LIGHT_GREY, self.c2p((100, 35 - self.car_width / 2)),
        #              self.c2p((-50, 35 - self.car_width / 2)), self.car_image[1].get_size()[1] * 2)

        # pg.draw.line(self.screen, (0, 0, 0), self.c2p((35 + self.car_width / 2, -50)),
        #              self.c2p((35 + self.car_width / 2, 100)), 1)
        # pg.draw.line(self.screen, (0, 0, 0), self.c2p((100, 35 - self.car_width / 2)),
        #              self.c2p((-50, 35 - self.car_width / 2)), 1)
        # bound1 = [[-1, 1], None] #[xbound, ybound], xbound = [x_min, x_max]  what is the meaning of bound?
        # bound2 = [None, [-1, 1]]
        # bound_set = [[[-12.5, 12.5], None], [None, [-1, 1]]]
        # for a in bound_set:
        #     bound_x, bound_y = a[0], a[1] #vehicle width
        #     if bound_x:
        #         b_min, b_max = bound_x[0], bound_x[1]
        #         _bound1 = self.c2p((b_min, 25))
        #         _bound2 = self.c2p((b_max, 25))
        #         bounds = np.array([_bound1[0], _bound2[0]])
        #         pg.draw.line(self.screen, LIGHT_GREY, ((bounds[1] + bounds[0])/2, -50),
        #                      ((bounds[1] + bounds[0])/2, self.screen_height * self.coordinate_scale,
        #                       ), bounds[1] - bounds[0])
        #
        #     if bound_y:
        #         b_min, b_max = bound_y[0], bound_y[1]
        #         _bound1 = self.c2p((25, b_min))
        #         _bound2 = self.c2p((25, b_max))
        #         bounds = np.array([_bound1[1], _bound2[1]])
        #         # pg.draw.line(self.screen, LIGHT_GREY, (0, (self.screen_width * self.coordinate_scale,
        #         #                 (bounds[1] + bounds[0]) / 2),
        #         #                 (bounds[1] + bounds[0]) / 2), bounds[0] - bounds[1])
        #
        #         pg.draw.line(self.screen, LIGHT_GREY, (self.screen_width * self.coordinate_scale,
        #                         (bounds[1] + bounds[0]) / 2), (-50, (bounds[1] + bounds[0]) / 2),
        #                         bounds[0] - bounds[1])

    def draw_dashed_line1(self):
        origin = self.c2p((35 + self.car_width, -50))
        target = self.c2p((35 + self.car_width, 100))
        displacement = target - origin
        length = abs(displacement[1])
        slope = displacement / length
        dash_length = 10

        for index in range(0, int(length / dash_length), 2):
            start = origin + (slope * index * dash_length)
            end = origin + (slope * (index + 1) * dash_length)
            pg.draw.line(self.screen, (0, 0, 0), start, end, 1)

    def draw_dashed_line2(self):
        origin = self.c2p((100, 35 - self.car_width))
        target = self.c2p((-50, 35 - self.car_width))
        displacement = target - origin
        length = abs(displacement[0])
        slope = displacement / length
        dash_length = 10

        for index in range(0, int(length / dash_length), 2):
            start = origin + (slope * index * dash_length)
            end = origin + (slope * (index + 1) * dash_length)
            pg.draw.line(self.screen, (0, 0, 0), start, end, 1)

    def c2p(self, coordinates):
        '''coordinates = x, y position in your environment(vehicle position)'''
        x = self.coordinate_scale * (- coordinates[0] + self.origin[0] + self.screen_width / 2)
        y = self.coordinate_scale * (- coordinates[1] + self.origin[1] + self.screen_height / 2)

        x = int(
            (x - self.screen_width * self.coordinate_scale * 0.5) * self.zoom
            + self.screen_width * self.coordinate_scale * 0.5)
        y = int(
            (y - self.screen_height * self.coordinate_scale * 0.5) * self.zoom
            + self.screen_height * self.coordinate_scale * 0.5)
        '''returns x, y for the pygame window'''

        return np.array([x, y])

    def generate(self, data):
        t_bar = np.arange(0, 3, 0.1)  # t_bar = 0: 0.1: 10
        X_bar = np.zeros((4, t_bar.shape[0]))
        i = 0
        j = 0
        time = 0
        t = data['t']  # time is from train_data
        X = data['X']
        while time <= 3.0:
            while t[0][i] <= time:
                i += 1
            """
            agent 1: original state: (py, px)
            agent 2: original state: (px, py)
            """
            X_bar[0][j] = (time - t[0][i - 1]) * (X[1][i] - X[1][i - 1]) / (t[0][i] - t[0][i - 1]) + X[1][i - 1]
            X_bar[1][j] = (time - t[0][i - 1]) * (X[0][i] - X[0][i - 1]) / (t[0][i] - t[0][i - 1]) + X[0][i - 1]
            X_bar[2][j] = (time - t[0][i - 1]) * (X[4][i] - X[4][i - 1]) / (t[0][i] - t[0][i - 1]) + X[4][i - 1]
            X_bar[3][j] = (time - t[0][i - 1]) * (X[5][i] - X[5][i - 1]) / (t[0][i] - t[0][i - 1]) + X[5][i - 1]
            time = time + 0.1
            j += 1

        new_data = dict()
        new_data.update({'t': t_bar,
                         'X': X_bar})

        return new_data

if __name__ == '__main__':
    vis = VisUtils()
    vis.draw_frame()

    # path = 'image_recording/'
    # import glob
    #
    # image = glob.glob(path + "*.png")
    # # print(image)
    # # episode_step_count = len(image)
    # img_list = image[-1] # [path + "img" + str(i).zfill(3) + ".png" for i in range(episode_step_count)]
    #
    #
    # import imageio
    #
    #
    # # # tag = 'E_E' + '_' + 'theta1' + '=' + 'a' + '_' + 'theta2' + '=' + 'na' + '_' + 'time horizon' + '=' + str(config.t1)
    # # tag = 'NE_E' + '_' + 'theta1' + '=' + 'a' + '_' + 'theta2' + '=' + 'na' + '_' + 'time horizon' + '=' + str(config.t1)
    # imageio.imsave(path + 'final_ ', img_list)
    # # imageio.mimsave(path + 'movie_' + tag + '.gif', images, 'GIF', fps=5)
    # # # Delete images
    # [os.remove(path + file) for file in os.listdir(path) if ".png" in file]
