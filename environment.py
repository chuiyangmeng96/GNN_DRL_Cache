import numpy as np
import gym


class Environ:
    def __init__(self):
        self.action_space = None
        self.length = 500
        self.eps = pow(10, -8)
        self.vehicle_num = 100
        self.RSU_num = 4
        self.RSUposition = np.array([[250, 0], [-250, 0], [0, -250], [0, 250]])
        self.MBSposition = np.array([0, 0])
        self.delta_int = 0.002
        self.T_wait = 2
        self.P_wait = 0.8
        self.dis_thresh = 100   # threshold of distance for different content request probability
        self.v = 60 / 3.6  # velocity m/s
        self.time_delta = 1  # time step
        self.gammaV = 100  # V2V threshold
        self.gammaR = 560  # V2R threshold
        self.z = None   # popularity for each content
        self.z1 = 3  # popularity for inner circle vehicles
        self.z2 = 10   # popularity for outer circle vehicles
        self.z3 = 5   # popularity for all RSUs'
        self.content_num = 50  # content number
        self.max_cacheV = 600  # total 20 vehicles caching capability 12000MB = 12GB, each 600MB, max 3 contents
        self.max_cacheR = 2e3  # total 4 RSUs caching capability 8000MB = 8GB, each 2GB, max 10 contents
        self.content_size = 200  # each content size = 200MB
        self.request_arr_V = np.zeros((self.vehicle_num, self.content_num))   # request array for each vehicle and each file
        self.request_arr_R = np.zeros((self.RSU_num, self.content_num))   # request array for each RSU and each file
        self.content_label_V = np.zeros((self.vehicle_num, int(self.max_cacheV / self.content_size)))   # vehicle caching content array
        self.content_label_R = np.zeros((self.vehicle_num, int(self.max_cacheR / self.content_size)))   # RSU caching content array
        self.candidateV = np.zeros((self.vehicle_num, self.vehicle_num))   # matrix stores dis between vehicles
        self.candidateR = np.zeros((self.vehicle_num, self.RSU_num))   # matrix stores dis between vehicles and RSUs
        # self.loop = 0 # need modification

        self.band_V2V = 5e6  # bandwidth 5MHz
        self.p_k = pow(10, 24 / 10)  # transmission power 24dBm to mW
        self.sigma_2 = pow(10, -11)  # noise power mW
        self.alpha = 2  # path loss exponent
        self.h_ik = 0.16  # channel gain  都是0.16

        self.band_V2R = 1e7  # bandwidth 10MHz
        self.bij = 1e7  # allocated bandwidth
        self.p_j = pow(10, 36 / 10)  # transmission power 36dBm to mW
        self.h_ij = 0.16  # channel gain
        # self.h_ij_total =    # total channel gain  不考虑干扰问题

        self.band_V2I = 2e7  # V2I bandwidth 20MHz
        self.W = 10  # number of sub-carriers
        self.p_m = pow(10, 43 / 10)  # transmission power 43dBM to mW
        self.h_i0 = 0.16  # channel gain

        self.position = np.zeros((self.vehicle_num, 4))

    def add_vehicle_pos(self):
        position = np.zeros((self.vehicle_num, 4))
        for i in range(self.vehicle_num):
            ind = np.random.rand(1)
            if ind < 1/12:
                position[i][0] = np.random.randint(-self.length + 1, 0)
                position[i][1] = self.length - 1
            elif 1/12 < ind < 2/12:
                position[i][0] = np.random.randint(0, self.length)
                position[i][1] = self.length - 1
            elif 2/12 < ind < 3/12:
                position[i][0] = -(self.length - 1)
                position[i][1] = np.random.randint(0, self.length)
            elif 3/12 < ind < 4/12:
                position[i][0] = 0
                position[i][1] = np.random.randint(0, self.length)
            elif 4/12 < ind < 5/12:
                position[i][0] = self.length - 1
                position[i][1] = np.random.randint(0, self.length)
            elif 5/12 < ind < 6/12:
                position[i][0] = np.random.randint(-self.length + 1, 0)
                position[i][1] = 0
            elif 6/12 < ind < 7/12:
                position[i][0] = np.random.randint(0, self.length)
                position[i][1] = 0
            elif 7/12 < ind < 8/12:
                position[i][0] = -(self.length - 1)
                position[i][1] = np.random.randint(-self.length + 1, 0)
            elif 8/12 < ind < 9/12:
                position[i][0] = 0
                position[i][1] = np.random.randint(-self.length + 1, 0)
            elif 9/12 < ind < 10/12:
                position[i][0] = self.length - 1
                position[i][1] = np.random.randint(-self.length + 1, 0)
            elif 10/12 < ind < 11/12:
                position[i][0] = np.random.randint(-self.length + 1, 0)
                position[i][1] = -(self.length - 1)
            else:
                position[i][0] = np.random.randint(0, self.length)
                position[i][1] = -(self.length - 1)

        self.position = position

    def update_position(self, obs_step):
        P_eta = 2 / (2 + self.T_wait * self.P_wait * self.delta_int * self.v) # moving probability
        for i in range(self.vehicle_num):
            if self.position[i][3] == obs_step:

                # initial EAST
                # head EAST with intersection at y=0 axis
                if self.position[i][2] == 0 and self.position[i][1] == 0 and self.position[i][0] * (
                        self.position[i][0] + self.time_delta * self.v) < 0:
                    stop_dice = np.random.rand(1)
                    if stop_dice < P_eta:  # move at intersection
                        self.position[i][3] = obs_step
                        direct_dice = np.random.rand(1)
                        if direct_dice < 0.25:  # proceed EAST
                            self.position[i][2] = 0
                            self.position[i][0] = self.position[i][0] + self.time_delta * self.v
                        elif 0.25 < direct_dice < 0.5:  # change WEST
                            self.position[i][2] = 1
                            self.position[i][0] = -(self.position[i][0] + self.time_delta * self.v)
                        elif 0.5 < direct_dice < 0.75:  # change SOUTH
                            self.position[i][2] = 2
                            self.position[i][1] = -(self.position[i][0] + self.time_delta * self.v)
                            self.position[i][0] = 0
                        else:  # change NORTH
                            self.position[i][2] = 3
                            self.position[i][1] = self.position[i][0] + self.time_delta * self.v
                            self.position[i][0] = 0
                    else:  # stop at intersection
                        self.position[i][0] = 0
                        self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head EAST no intersection at y=0 axis
                elif self.position[i][2] == 0 and self.position[i][1] == 0 and self.position[i][0] * (
                        self.position[i][0] + self.time_delta * self.v) > 0:
                    self.position[i][3] = obs_step
                    self.position[i][0] = self.position[i][0] + self.time_delta * self.v
                    if self.position[i][0] > self.length - 1:  # out of bound ### NEED MODIFICATION
                        stop_dice = np.random.rand(1)
                        if stop_dice < P_eta:  # move at intersection
                            direct_dice = np.random.rand(1)
                            if direct_dice < 1/3: # U turn WEST
                                self.position[i][2] = 1
                                self.position[i][0] = 2 * (self.length - 1) - self.position[i][0]
                            elif 1/3 < direct_dice < 2/3: # change NORTH
                                self.position[i][2] = 3
                                self.position[i][1] = self.position[i][0] - (self.length - 1)
                                self.position[i][0] = self.length - 1
                            else: # change SOUTH
                                self.position[i][2] = 2
                                self.position[i][1] = (self.length - 1) - self.position[i][0]
                                self.position[i][0] = self.length - 1
                        else: # stop at intersection
                            self.position[i][0] = self.length - 1
                            self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head EAST with intersection at y = self.length-1 axis
                elif self.position[i][2] == 0 and self.position[i][1] == (self.length - 1) and self.position[i][0] * (
                        self.position[i][0] + self.time_delta * self.v) < 0:
                    stop_dice = np.random.rand(1)
                    if stop_dice < P_eta:  # move at intersection
                        self.position[i][3] = obs_step
                        direct_dice = np.random.rand(1)
                        if direct_dice < 1/3:  # proceed EAST
                            self.position[i][2] = 0
                            self.position[i][0] = self.position[i][0] + self.time_delta * self.v
                        elif 1/3 < direct_dice < 2/3:  # change WEST
                            self.position[i][2] = 1
                            self.position[i][0] = -(self.position[i][0] + self.time_delta * self.v)
                        else:   # change SOUTH
                            self.position[i][2] = 2
                            self.position[i][1] = (self.length - 1) - (self.position[i][0] + self.time_delta * self.v)
                            self.position[i][0] = 0
                    else:  # stop at intersection
                        self.position[i][0] = 0
                        self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head EAST no intersection at y = self.length - 1 axis
                elif self.position[i][2] == 0 and self.position[i][1] == (self.length - 1) and self.position[i][0] * (
                        self.position[i][0] + self.time_delta * self.v) > 0:
                    self.position[i][3] = obs_step
                    self.position[i][0] = self.position[i][0] + self.time_delta * self.v
                    if self.position[i][0] > self.length - 1:  # out of bound
                        stop_dice = np.random.rand(1)
                        if stop_dice < P_eta:   # move at CORNER
                            direct_dice = np.random.rand(1)
                            if direct_dice < 1/2: # U turn WEST
                                self.position[i][2] = 1
                                self.position[i][0] = 2 * (self.length - 1) - self.position[i][0]
                            else: # change SOUTH
                                self.position[i][2] = 2
                                self.position[i][1] = 2 * (self.length - 1) - self.position[i][0]
                                self.position[i][0] = self.length - 1
                        else:  # STOP at CORNER
                            self.position[i][0] = self.length - 1
                            self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head EAST with intersection at y = -(self.length-1) axis
                elif self.position[i][2] == 0 and self.position[i][1] == -(self.length - 1) and self.position[i][0] * (
                        self.position[i][0] + self.time_delta * self.v) < 0:
                    stop_dice = np.random.rand(1)
                    if stop_dice < P_eta:  # move at intersection
                        self.position[i][3] = obs_step
                        direct_dice = np.random.rand(1)
                        if direct_dice < 1 / 3:  # proceed EAST
                            self.position[i][2] = 0
                            self.position[i][0] = self.position[i][0] + self.time_delta * self.v
                        elif 1 / 3 < direct_dice < 2 / 3:  # change WEST
                            self.position[i][2] = 1
                            self.position[i][0] = -(self.position[i][0] + self.time_delta * self.v)
                        else:  # change NORTH
                            self.position[i][2] = 2
                            self.position[i][1] = -(self.length - 1) + (self.position[i][0] + self.time_delta * self.v)
                            self.position[i][0] = 0
                    else:  # stop at intersection
                        self.position[i][0] = 0
                        self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head EAST no intersection at y = -(self.length-1) axis
                elif self.position[i][2] == 0 and self.position[i][1] == -(self.length - 1) and self.position[i][0] * (
                        self.position[i][0] + self.time_delta * self.v) > 0:
                    self.position[i][3] = obs_step
                    self.position[i][0] = self.position[i][0] + self.time_delta * self.v
                    if self.position[i][0] > self.length - 1:  # out of bound
                        stop_dice = np.random.rand(1)
                        if stop_dice < P_eta:   # move at CORNER
                            direct_dice = np.random.rand(1)
                            if direct_dice < 1/2: # U turn WEST
                                self.position[i][2] = 1
                                self.position[i][0] = 2 * (self.length - 1) - self.position[i][0]
                            else: # change NORTH
                                self.position[i][2] = 3
                                self.position[i][1] = -2 * (self.length - 1) + self.position[i][0]
                                self.position[i][0] = self.length - 1
                        else:  # STOP at CORNER
                            self.position[i][0] = self.length - 1
                            self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # intial WEST
                # head WEST with intersection at y = 0 axis
                if self.position[i][2] == 1 and self.position[i][1] == 0 and self.position[i][0] * (
                        self.position[i][0] + self.time_delta * self.v) < 0:
                    stop_dice = np.random.rand(1)
                    if stop_dice < P_eta:  # move at intersection
                        self.position[i][3] = obs_step
                        direct_dice = np.random.rand(1)
                        if direct_dice < 0.25:  # change EAST
                            self.position[i][2] = 0
                            self.position[i][0] = -self.position[i][0] + self.time_delta * self.v
                        elif 0.25 < direct_dice < 0.5:  # proceed WEST
                            self.position[i][2] = 1
                            self.position[i][0] = self.position[i][0] - self.time_delta * self.v
                        elif 0.5 < direct_dice < 0.75:  # change SOUTH
                            self.position[i][2] = 2
                            self.position[i][1] = self.position[i][0] - self.time_delta * self.v
                            self.position[i][0] = 0
                        else:  # change NORTH
                            self.position[i][2] = 3
                            self.position[i][1] = -self.position[i][0] + self.time_delta * self.v
                            self.position[i][0] = 0
                    else:  # stop at intersection
                        self.position[i][0] = 0
                        self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head WEST no intersection at y = 0 axis
                elif self.position[i][2] == 1 and self.position[i][1] == 0 and self.position[i][0] * (
                        self.position[i][0] + self.time_delta * self.v) > 0:
                    self.position[i][3] = obs_step
                    self.position[i][0] = self.position[i][0] - self.time_delta * self.v
                    if self.position[i][0] < -(self.length - 1):  # out of bound
                        stop_dice = np.random.rand(1)
                        if stop_dice < P_eta: # move at intersection
                            direct_dice = np.random.rand(1)
                            if direct_dice < 1/3: # U turn EAST
                                self.position[i][2] = 0
                                self.position[i][0] = 2 * (-self.length + 1) - self.position[i][0]
                            elif 1/3 < direct_dice < 2/3: # change SOUTH
                                self.position[i][2] = 2
                                self.position[i][1] = self.position[i][0] + (self.length - 1)
                                self.position[i][0] = -(self.length - 1)
                            else: # change NORTH
                                self.position[i][2] = 3
                                self.position[i][1] = -self.position[i][0] - (self.length - 1)
                                self.position[i][0] = -(self.length - 1)
                        else: # STOP at intersection
                            self.position[i][0] = -(self.length - 1)
                            self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head WEST with intersection at y = self.length - 1 axis
                elif self.position[i][2] == 1 and self.position[i][1] == self.length - 1 and self.position[i][0] * (
                        self.position[i][0] + self.time_delta * self.v) < 0:
                    stop_dice = np.random.rand(1)
                    if stop_dice < P_eta:  # move at intersection
                        self.position[i][3] = obs_step
                        direct_dice = np.random.rand(1)
                        if direct_dice < 1/3:  # change EAST
                            self.position[i][2] = 0
                            self.position[i][0] = -self.position[i][0] + self.time_delta * self.v
                        elif 0.25 < direct_dice < 0.5:  # proceed WEST
                            self.position[i][2] = 1
                            self.position[i][0] = self.position[i][0] - self.time_delta * self.v
                        elif 0.5 < direct_dice < 0.75:  # change SOUTH
                            self.position[i][2] = 2
                            self.position[i][1] = (self.length - 1) + self.position[i][0] - self.time_delta * self.v
                            self.position[i][0] = 0
                    else:  # stop at intersection
                        self.position[i][0] = 0
                        self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head WEST no intersection at y = self.length - 1 axis
                elif self.position[i][2] == 1 and self.position[i][1] == self.length - 1 and self.position[i][0] * (
                        self.position[i][0] + self.time_delta * self.v) > 0:
                    self.position[i][3] = obs_step
                    self.position[i][0] = self.position[i][0] - self.time_delta * self.v
                    if self.position[i][0] < -(self.length - 1):  # out of bound at CORNER
                        stop_dice = np.random.rand(1)
                        if stop_dice < P_eta: # move at intersection
                            direct_dice = np.random.rand(1)
                            if direct_dice < 1/2: # U turn EAST
                                self.position[i][2] = 0
                                self.position[i][0] = 2 * (-self.length + 1) - self.position[i][0]
                            else: # change SOUTH
                                self.position[i][2] = 2
                                self.position[i][1] = self.position[i][0] + 2 * (self.length - 1)
                                self.position[i][0] = -(self.length - 1)
                        else: # STOP at CORNER
                            self.position[i][0] = -(self.length - 1)
                            self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head WEST with intersection at y = -(self.length - 1) axis
                elif self.position[i][2] == 1 and self.position[i][1] == -(self.length - 1) and self.position[i][0] * (
                        self.position[i][0] + self.time_delta * self.v) < 0:
                    stop_dice = np.random.rand(1)
                    if stop_dice < P_eta:  # move at intersection
                        self.position[i][3] = obs_step
                        direct_dice = np.random.rand(1)
                        if direct_dice < 1/3: # change EAST
                            self.position[i][2] = 0
                            self.position[i][0] = -self.position[i][0] + self.time_delta * self.v
                        elif 1/3 < direct_dice < 2/3: # proceed WEST
                            self.position[i][2] = 1
                            self.position[i][0] = self.position[i][0] - self.time_delta * self.v
                        else: # change NORTH
                            self.position[i][2] = 3
                            self.position[i][1] = -(self.length - 1) + self.position[i][0] - self.time_delta * self.v
                            self.position[i][0] = 0
                    else: # stop at intersection
                        self.position[i][0] = 0
                        self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head WEST no intersection at y = -(self.length - 1) axis
                elif self.position[i][2] == 1 and self.position[i][1] == -(self.length - 1) and self.position[i][0] * (
                        self.position[i][0] + self.time_delta * self.v) > 0:
                    self.position[i][3] = obs_step
                    self.position[i][0] = self.position[i][0] - self.time_delta * self.v
                    if self.position[i][0] < -(self.length - 1):  # out of bound at CORNER
                        stop_dice = np.random.rand(1)
                        if stop_dice < P_eta: # move at intersection
                            direct_dice = np.random.rand(1)
                            if direct_dice < 1/2: # U turn EAST
                                self.position[i][2] = 0
                                self.position[i][0] = 2 * (-self.length + 1) - self.position[i][0]
                            else: # change NORTH
                                self.position[i][2] = 3
                                self.position[i][1] = 2 * (-self.length + 1) + self.position[i][0]
                                self.position[i][0] = -(self.length - 1)
                        else: # STOP at CORNER
                            self.position[i][0] = -(self.length - 1)
                            self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # intial SOUTH
                # head SOUTH with intersection at x = 0 axis
                if self.position[i][2] == 2 and self.position[i][0] == 0 and self.position[i][1] * (
                        self.position[i][1]+ self.time_delta * self.v) < 0:
                    stop_dice = np.random.rand(1)
                    if stop_dice < P_eta:  # move at intersection
                        self.position[i][3] = obs_step
                        direct_dice = np.random.rand(1)
                        if direct_dice < 0.25:  # change EAST
                            self.position[i][2] = 0
                            self.position[i][0] = -self.position[i][1] + self.time_delta * self.v
                            self.position[i][1] = 0
                        elif 0.25 < direct_dice < 0.5:  # change WEST
                            self.position[i][2] = 1
                            self.position[i][0] = self.position[i][0] - self.time_delta * self.v
                            self.position[i][1] = 0
                        elif 0.5 < direct_dice < 0.75:  # proceed SOUTH
                            self.position[i][2] = 2
                            self.position[i][1] = self.position[i][1] - self.time_delta * self.v
                        else:  # change NORTH
                            self.position[i][2] = 3
                            self.position[i][1] = -self.position[i][1] + self.time_delta * self.v
                    else:  # stop at intersection
                        self.position[i][1] = 0
                        self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head SOUTH no intersection at x = 0 axis
                elif self.position[i][2] == 2 and self.position[i][0] == 0 and self.position[i][1] * (
                        self.position[i][1] + self.time_delta * self.v) > 0:
                    self.position[i][3] = obs_step
                    self.position[i][1] = self.position[i][1] - self.time_delta * self.v
                    if self.position[i][1] < -self.length + 1:  # out of bound
                        stop_dice = np.random.rand(1)
                        if stop_dice < P_eta: # move at intersection
                            direct_dice = np.random.rand(1)
                            if direct_dice < 1/3: # change EAST
                                self.position[i][2] = 0
                                self.position[i][0] = -(self.length - 1) - self.position[i][1]
                                self.position[i][1] = -(self.length - 1)
                            elif 1/3 < direct_dice < 2/3: # change SOUTH
                                self.position[i][2] = 2
                                self.position[i][0] = (self.length - 1) + self.position[i][1]
                                self.position[i][1] = -(self.length - 1)
                            else: # U turn to NORTH
                                self.position[i][2] = 3
                                self.position[i][1] = 2 * (-self.length + 1) - self.position[i][1]
                        else: # stop at intersection
                            self.position[i][1] = -(self.length - 1)
                            self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head SOUTH with intersection at x = self.length - 1 axis
                elif self.position[i][2] == 2 and self.position[i][0] == self.length - 1 and self.position[i][1] * (
                        self.position[i][1] + self.time_delta * self.v) < 0:
                    stop_dice = np.random.rand(1)
                    if stop_dice < P_eta:  # move at intersection
                        self.position[i][3] = obs_step
                        direct_dice = np.random.rand(1)
                        if direct_dice < 1/3:  # change WEST
                            self.position[i][2] = 1
                            self.position[i][0] = (self.length - 1) + self.position[i][0] - self.time_delta * self.v
                            self.position[i][1] = 0
                        elif 1/3 < direct_dice < 2/3:  # proceed SOUTH
                            self.position[i][2] = 2
                            self.position[i][1] = self.position[i][1] - self.time_delta * self.v
                        else:  # change NORTH
                            self.position[i][2] = 3
                            self.position[i][1] = -self.position[i][1] + self.time_delta * self.v
                    else:  # stop at intersection
                        self.position[i][1] = 0
                        self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head SOUTH no intersection at x = self.length - 1 axis
                elif self.position[i][2] == 2 and self.position[i][0] == self.length - 1 and self.position[i][1] * (
                        self.position[i][1] + self.time_delta * self.v) > 0:
                    self.position[i][3] = obs_step
                    self.position[i][1] = self.position[i][1] - self.time_delta * self.v
                    if self.position[i][1] < -self.length + 1:  # out of bound at corner
                        stop_dice = np.random.rand(1)
                        if stop_dice < P_eta:  # move at intersection
                            direct_dice = np.random.rand(1)
                            if direct_dice < 0.5: # change WEST
                                self.position[i][2] = 1
                                self.position[i][0] = 2 * (self.length - 1) + self.position[i][1]
                                self.position[i][1] = -(self.length - 1)
                            else: # U turn to NORTH
                                self.position[i][2] = 3
                                self.position[i][1] = 2 * (-self.length + 1) - self.position[i][1]

                # head SOUTH with intersection at x = -(self.length - 1) axis
                elif self.position[i][2] == 2 and self.position[i][0] == -(self.length - 1) and self.position[i][1] * (
                        self.position[i][1] + self.time_delta * self.v) < 0:
                    stop_dice = np.random.rand(1)
                    if stop_dice < P_eta:  # move at intersection
                        self.position[i][3] = obs_step
                        direct_dice = np.random.rand(1)
                        if direct_dice < 1 / 3:  # change EAST
                            self.position[i][2] = 0
                            self.position[i][0] = -(self.length - 1) - (self.position[i][0] - self.time_delta * self.v)
                            self.position[i][1] = 0
                        elif 1 / 3 < direct_dice < 2 / 3:  # proceed SOUTH
                            self.position[i][2] = 2
                            self.position[i][1] = self.position[i][1] - self.time_delta * self.v
                        else:  # change NORTH
                            self.position[i][2] = 3
                            self.position[i][1] = -self.position[i][1] + self.time_delta * self.v
                    else:  # stop at intersection
                        self.position[i][1] = 0
                        self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head SOUTH no intersection at x = -(self.length - 1) axis
                elif self.position[i][2] == 2 and self.position[i][0] == -(self.length - 1) and self.position[i][1] * (
                        self.position[i][1] + self.time_delta * self.v) > 0:
                    self.position[i][3] = obs_step
                    self.position[i][1] = self.position[i][1] - self.time_delta * self.v
                    if self.position[i][1] < -self.length + 1:  # out of bound at corner
                        stop_dice = np.random.rand(1)
                        if stop_dice < P_eta:  # move at intersection
                            direct_dice = np.random.rand(1)
                            if direct_dice < 0.5: # change EAST
                                self.position[i][2] = 0
                                self.position[i][0] = 2 * (-self.length + 1) - self.position[i][1]
                                self.position[i][1] = -(self.length - 1)
                            else: # U turn to NORTH
                                self.position[i][2] = 3
                                self.position[i][1] = 2 * (-self.length + 1) - self.position[i][1]

                # intial NORTH
                # head NORTH with intersection at x = 0 axis
                if self.position[i][2] == 3 and self.position[i][0] == 0 and self.position[i][1] * (
                        self.position[i][1] + self.time_delta * self.v) < 0:
                    stop_dice = np.random.rand(1)
                    if stop_dice < P_eta:  # move at intersection
                        self.position[i][3] = obs_step
                        direct_dice = np.random.rand(1)
                        if direct_dice < 0.25:  # change EAST
                            self.position[i][2] = 0
                            self.position[i][0] = self.position[i][1] + self.time_delta * self.v
                            self.position[i][1] = 0
                        elif 0.25 < direct_dice < 0.5:  # change WEST
                            self.position[i][2] = 1
                            self.position[i][0] = -(self.position[i][0] + self.time_delta * self.v)
                            self.position[i][1] = 0
                        elif 0.5 < direct_dice < 0.75:  # change SOUTH
                            self.position[i][2] = 2
                            self.position[i][1] = -(self.position[i][1] + self.time_delta * self.v)
                        else:  # proceed NORTH
                            self.position[i][2] = 3
                            self.position[i][1] = self.position[i][1] + self.time_delta * self.v
                    else:  # stop at intersection
                        self.position[i][1] = 0
                        self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head NORTH no intersection at x = 0 axis
                elif self.position[i][2] == 3 and self.position[i][0] == 0 and self.position[i][1] * (
                        self.position[i][1] + self.time_delta * self.v) > 0:
                    self.position[i][3] = obs_step
                    self.position[i][1] = self.position[i][1] + self.time_delta * self.v
                    if self.position[i][1] > self.length - 1:  # out of bound
                        stop_dice = np.random.rand(1)
                        if stop_dice < P_eta:  # move at intersection
                            direct_dice = np.random.rand(1)
                            if direct_dice < 1/3: # change EAST
                                self.position[i][2] = 0
                                self.position[i][0] = self.position[i][1] - (self.length - 1)
                                self.position[i][1] = self.length - 1
                            elif 1/3 < direct_dice < 2/3: # change WEST
                                self.position[i][2] = 1
                                self.position[i][0] = -self.position[i][1] + (self.length - 1)
                                self.position[i][1] = self.length - 1
                            else: # U turn to SOUTH
                                self.position[i][2] = 2
                                self.position[i][1] = 2 * (self.length - 1) - self.position[i][1]
                        else: # stop at intersection
                            self.position[i][1] = self.length - 1
                            self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head NORTH with intersection at x = self.length - 1 axis
                elif self.position[i][2] == 3 and self.position[i][0] == self.length - 1 and self.position[i][1] * (
                        self.position[i][1] + self.time_delta * self.v) < 0:
                    stop_dice = np.random.rand(1)
                    if stop_dice < P_eta:  # move at intersection
                        self.position[i][3] = obs_step
                        direct_dice = np.random.rand(1)
                        if direct_dice < 1 / 3:  # change WEST
                            self.position[i][2] = 1
                            self.position[i][0] = (self.length - 1) - self.time_delta * self.v
                            self.position[i][1] = 0
                        elif 1 / 3 < direct_dice < 2 / 3: # change SOUTH
                            self.position[i][2] = 2
                            self.position[i][1] = -(self.position[i][1] + self.time_delta * self.v)
                        else: # proceed NORTH
                            self.position[i][2] = 3
                            self.position[i][1] = self.position[i][1] + self.time_delta * self.v
                    else:
                        self.position[i][1] = 0
                        self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head NORTH no intersection at x = self.length - 1 axis
                elif self.position[i][2] == 3 and self.position[i][0] == self.length - 1 and self.position[i][1] * (
                        self.position[i][1] + self.time_delta * self.v) > 0:
                    self.position[i][3] = obs_step
                    self.position[i][1] = self.position[i][1] + self.time_delta * self.v
                    if self.position[i][1] > self.length - 1:  # out of bound at CORNER
                        stop_dice = np.random.rand(1)
                        if stop_dice < P_eta:  # move at intersection
                            direct_dice = np.random.rand(1)
                            if direct_dice < 1/2: # change WEST
                                self.position[i][2] = 1
                                self.position[i][0] = 2 * (self.length - 1) - self.position[i][1]
                                self.position[i][1] = self.length - 1
                            else: # U turn to SOUTH
                                self.position[i][2] = 2
                                self.position[i][1] = 2 * (self.length - 1) - self.position[i][1]
                        else: # stop at intersection
                            self.position[i][1] = self.length - 1
                            self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head NORTH with intersection at x = -(self.length - 1) axis
                elif self.position[i][2] == 3 and self.position[i][0] == -(self.length - 1) and self.position[i][1] * (
                        self.position[i][1] + self.time_delta * self.v) < 0:
                    stop_dice = np.random.rand(1)
                    if stop_dice < P_eta:  # move at intersection
                        self.position[i][3] = obs_step
                        direct_dice = np.random.rand(1)
                        if direct_dice < 1 / 3:  # change EAST
                            self.position[i][2] = 0
                            self.position[i][0] = -(self.length - 1) + self.time_delta * self.v
                            self.position[i][1] = 0
                        elif 1 / 3 < direct_dice < 2 / 3: # change SOUTH
                            self.position[i][2] = 2
                            self.position[i][1] = -(self.position[i][1] + self.time_delta * self.v)
                        else: # proceed NORTH
                            self.position[i][2] = 3
                            self.position[i][1] = self.position[i][1] + self.time_delta * self.v
                    else:
                        self.position[i][1] = 0
                        self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

                # head NORTH no intersection at x = self.length - 1 axis
                elif self.position[i][2] == 3 and self.position[i][0] == -(self.length - 1) and self.position[i][1] * (
                        self.position[i][1] + self.time_delta * self.v) > 0:
                    self.position[i][3] = obs_step
                    self.position[i][1] = self.position[i][1] + self.time_delta * self.v
                    if self.position[i][1] > self.length - 1:  # out of bound at CORNER
                        stop_dice = np.random.rand(1)
                        if stop_dice < P_eta:  # move at intersection
                            direct_dice = np.random.rand(1)
                            if direct_dice < 1/2: # change EAST
                                self.position[i][2] = 0
                                self.position[i][0] = -2 * (self.length - 1) + self.position[i][1]
                                self.position[i][1] = self.length - 1
                            else: # U turn to SOUTH
                                self.position[i][2] = 2
                                self.position[i][1] = 2 * (self.length - 1) - self.position[i][1]
                        else: # stop at intersection
                            self.position[i][1] = self.length - 1
                            self.position[i][3] = self.position[i][3] + 2 / 0.1  # 20 is average stop time/interval

        return self.position

    def cal_neighbor(self): # self,candidateV rewrite
        # V2V, V2R distance calculation
        for i in range(self.vehicle_num):
            for j in range(self.vehicle_num):
                temp_V = np.sqrt(np.sum(np.power(self.position[i, 0:2] - self.position[j, 0:2], 2)))
                if i != j and temp_V < self.gammaV:
                    self.candidateV[i][j] = temp_V
            for k in range(self.RSU_num):
                temp_R = np.sqrt(np.sum(np.power(self.position[i, 0:2] - self.RSUposition[k, :], 2)))
                if temp_R < self.gammaR:
                    self.candidateR[i][k] = temp_R
        return self.candidateV, self.candidateR

    def content_request(self): # define different request probability in inner/outer circle for all Vehicles and RSUs
        for i in range(self.vehicle_num):   # for Vehicles
            temp_request_arr = np.zeros((1, self.content_num))
            if np.sqrt(np.sum(np.power(self.position[i, 0:2], 2))) < self.dis_thresh:   # inner circle
                self.z = self.z1
                temp_content_deno = np.zeros(self.content_num)
                for j in range(self.content_num):
                    temp_content_deno[j] = 1 / pow(j + 1, self.z)  # begin at 0
                content_deno = np.sum(temp_content_deno)
                for j in range(self.content_num):
                    temp_request_arr[0][j] = temp_content_deno[j] / content_deno   # begin at 0
                self.request_arr_V[i, :] = temp_request_arr
            else:   # outer circle
                self.z = self.z2
                temp_content_deno = np.zeros(self.content_num)
                for j in range(self.content_num):
                    temp_content_deno[j] = 1 / pow(j + 1, self.z)  # begin at 0
                content_deno = np.sum(temp_content_deno)
                for j in range(self.content_num):
                    temp_request_arr[0][j] = temp_content_deno[j] / content_deno   # begin at 0
                self.request_arr_V[i, :] = temp_request_arr

        for j in range(self.RSU_num):   # for RSUs
            temp_request_arr = np.zeros((1, self.content_num))
            self.z = self.z3
            temp_content_deno = np.zeros(self.content_num)
            for i in range(self.content_num):
                temp_content_deno[j] = 1 / pow(j + 1, self.z)   # begin at 0
            content_deno = np.sum(temp_content_deno)
            for k in range(self.content_num):
                temp_request_arr[0][j] = (1 / pow(k + 1, self.z)) / content_deno  # begin at 0
            self.request_arr_R[j, :] = temp_request_arr
        return self.request_arr_V, self.request_arr_R

    def caching_strategy(self): # define basic caching strategy with LRU
        for i in range(self.vehicle_num):  # Vehicle caching strategy
            dice = np.random.rand(1)
            if 0 < dice < self.request_arr_V[i][0]:   # randomly request a content and its label
                content_label = 0
            else:
                for j in range(1, self.content_num):
                    if np.sum(self.request_arr_V[i, 0:j - 1]) < dice < np.sum(self.request_arr_V[i, 0:j]):
                       content_label = j - 1

            if content_label in self.content_label_V[i, :]:   # case 1 requested content already in vehicle cache
                content_label_index = list(self.content_label_V[i, :]).index(content_label)
                if content_label_index != 0:
                    for j in range(content_label_index):
                        self.content_label_V[i][j + 1] = self.content_label_V[i][j]
                    self.content_label_V[i][0] = content_label
            else:  # requested content not in vehicle, enough/not enough (same case) space for storing new content
                for j in range(int(self.max_cacheV / self.content_size) - 1):
                    self.content_label_V[i][j + 1] = self.content_label_V[i][j]
                self.content_label_V[i][0] = content_label # self.content_label_V[i][1] = self.content_label_V[i][0]

        for i in range(self.RSU_num):
            dice = np.random.rand(1)
            if 0 < dice < self.request_arr_R[i][0]:  # randomly request a content and its label
                content_label = 0
            else:
                for j in range(1, self.content_num):
                    if np.sum(self.request_arr_R[i, 0:j - 1]) < dice < np.sum(self.request_arr_R[i, 0:j]):
                        content_label = j

            if content_label in self.content_label_R[i, :]:   # case 1 requested content already in RSU cache
                content_label_index = list(self.content_label_R[i, :]).index(content_label)
                if content_label_index != 0:
                    for j in range(content_label_index):
                        self.content_label_R[i][j + 1] = self.content_label_R[i][j]
                    self.content_label_R[i][0] = content_label
            else:   # requested content not in RSU
                # length = self.max_cacheR / self.content_size
                for j in range(int(self.max_cacheR / self.content_size) - 1):
                    self.content_label_R[i][j + 1] = self.content_label_R[i][j]
                self.content_label_R[i][0] = content_label
        return self.content_label_V, self.content_label_R

    def input(self):   # combine as a state matrix
        #   section 1: neighboring vehicles
        # global self, i, pos_dis, index_R
        input_1 = np.zeros((self.vehicle_num, self.vehicle_num))
        for i in range(self.vehicle_num):
            # connect_index_V = np.argwhere(self.candidateV[i, :] != 0)
            connect_index_V = np.nonzero(self.candidateV[i, :])[0]
            for index in connect_index_V:
                input_1[i][index] = 1

        #   section 2: transmission rate / delay
        input_2 = np.zeros((self.vehicle_num, self.vehicle_num + self.RSU_num + 1))
        for i in range(self.vehicle_num):
            # connect_index_V = np.argwhere(self.candidateV[i, :] != 0)
            connect_index_V = np.nonzero(self.candidateV[i, :])[0]
            for index_V in connect_index_V:
                input_2[i][index_V] = self.band_V2V * np.log2(1 + self.p_k * self.h_ik * pow(self.candidateV[i][index_V], -self.alpha) / self.sigma_2)
            # connect_index_R = np.argwhere(self.candidateR[i, :] != 0)
            connect_index_R = np.nonzero(self.candidateR[i, :])[0]
            for index_R in connect_index_R:
                input_2[i][self.vehicle_num + index_R] = self.bij * np.log2(1 + self.p_j * self.h_ij * pow(self.candidateR[i][index_R], -self.alpha) / self.sigma_2)
            pos_dis = np.sqrt(np.sum(pow(self.position[i, 0:2], 2)))
            input_2[i][self.vehicle_num + self.RSU_num] = self.band_V2I / self.W * np.log2(1 + self.p_m * self.h_i0 * pow(pos_dis, -self.alpha) / self.sigma_2)

        #   section 3: Vehicle cached contents
        input_3 = np.array((self.vehicle_num, self.vehicle_num * self.content_num))
        for i in range(self.vehicle_num):
            for j in self.content_label_V[i, :]: # store self cache information
                input_3[i][i * self.content_num + j] = 1
            # connect_index_V = np.argwhere(self.candidateV[i, :] != 0)
            connect_index_V = np.nonzero(self.candidateV[i, :])[0]
            for index in connect_index_V:   # store other vehicle information
                # index = self.candidateV[index][1]
                for j in self.content_label_V[index, :]:
                    input_3[i][index * self.content_num + j] = 1

        #   section 4: RSU cached contents
        input_4 = np.array((self.vehicle_num, self.RSU_num * self.content_num))
        for i in range(self.vehicle_num):
            # connect_index_R = np.argwhere(self.candidateR[:, 0] != 0)
            connect_index_R = np.nonzero(self.candidateR[i, :])[0]
            for index in connect_index_R:
                # index = self.candidateR[index][1]
                for j in self.content_label_R[index, :]:
                    input_4[i][index * self.content_num + j] = 1

        input_matrix = np.concatenate((input_1, input_2, input_3, input_4), axis=1)
        return  input_matrix

    def V2V_rate(self, position_A, position_B):  # achievable data rate
        d_ik = np.sqrt(np.sum(np.power(position_A - position_B, 2)))
        R_ki = self.band_V2V * np.log2(1 + self.p_k * self.h_ik * pow(d_ik + self.eps, -self.alpha) / self.sigma_2)/1e6
        return R_ki

    def V2V_time(self, y_ikf, position_A, position_B):  # delivery time
        R_ki = self.V2V_rate(position_A, position_B)
        return y_ikf * self.content_size / R_ki

    def V2R_rate(self, position_A, position_B):
        d_ij = np.sqrt(np.sum(np.power(position_A - position_B, 2)))
        R_ji = self.bij * np.log2(1 + self.p_j * self.h_ij * pow(d_ij + self.eps, -self.alpha) / self.sigma_2)/1e6
        return R_ji

    def V2R_time(self, y_ijf, position_A, position_B):  # delivery time
        R_ji = self.V2R_rate(position_A, position_B)
        return y_ijf * self.content_size / R_ji

    def V2I_rate(self, position_A):
        d_i0 = np.sqrt(np.sum(np.power(position_A - self.MBSposition, 2)))
        R_0i = (self.band_V2I / self.W * np.log2(1 + self.p_m * self.h_i0 * pow(d_i0 + self.eps, -self.alpha) / self.sigma_2))/1e6
        return R_0i

    def V2I_time(self, position_A):
        R_0i = self.V2I_rate(position_A)
        return self.content_size / R_0i   # no ratio

    def delay_func(self, action_matrix):  # x_if, x_jf need modification
        # y_ikf, y_ijf, y_i0f are ACTION SET which should be assigned afterwards
        V2V_delay = np.zeros((self.vehicle_num, self.vehicle_num))
        for num in range(self.vehicle_num):
            connect_num = np.nonzero(action_matrix[num, 0:self.vehicle_num])[0]
            for i in connect_num:
                y_ikf = action_matrix[num][i]
                position_A = self.position[num, 0:2]
                position_B = self.position[i, 0:2]
                T_ki = self.V2V_time(y_ikf, position_A, position_B)
                V2V_delay[num][i] = T_ki

        V2R_delay = np.zeros((self.vehicle_num, self.RSU_num))
        for num in range(self.vehicle_num):
            connect_num = np.nonzero(action_matrix[num, self.vehicle_num:self.vehicle_num + self.RSU_num])[0]
            for j in connect_num:
                y_ijf = action_matrix[num][self.vehicle_num + j]
                position_A = self.position[num, 0:2]
                position_B = self.RSUposition[j, 0:2]
                T_ji = self.V2R_time(y_ijf, position_A, position_B)
                V2R_delay[num][j] = T_ji

        V2I_delay = np.zeros((self.vehicle_num, 1))
        for k in range(self.vehicle_num):
            position_A = self.position[k, 0:2]
            V2I_delay[k] = self.V2I_time(position_A)

        total_delay = np.sum(V2V_delay, axis=1) + np.sum(V2R_delay, axis=1) + V2I_delay  # total_delay with all vehicles
        return total_delay

    def action_space(self): # action_matrix is a matrix of size self.vehicle_num * (self.vehicle_num + self.RSU_num + 1)
        action_matrix = np.random.rand(np.zeros(self.vehicle_num))
        return action_matrix

    def immediate_reward(self, action_matrix):
        total_delay = self.delay_func(action_matrix)
        reward_arr = np.zeros((self.vehicle_num, self.content_num))
        deno_arr = np.zeros(self.content_num)
        for j in range(self.content_num):
            deno_arr[j] = pow(j + 1, -self.z)
        deno = np.sum(deno_arr)

        for i in range(self.vehicle_num):
            for j in range(self.content_num):
                pf = pow(j + 1, -self.z) / deno
                reward_arr[i][j] = 20/(pf * total_delay[i])
        R = -np.sum(reward_arr)
        return R

    def step(self, action_matrix):  # action set size self.vehicle_num*(self.vehicle_num + self.RSU_num + 1)
        for i in range(self.vehicle_num):
            if action_matrix[i][self.vehicle_num + self.RSU_num] != 0:  # define extra penalty when receiving content from MBS
                r = -100
            else:
                r = 1 # need modification
        return r

    def packet_loss_func(self, total_delay): # change state by changing content distribution, T refers to delivery time of each vehicle (N*1)
        for i in range(self.vehicle_num):
            if total_delay[i] > self.time_delta:
                for j in range(int(self.max_cacheV / self.content_size) - 1):
                    self.content_label_V[i][j + 1] = self.content_label_V[i][j]
                self.content_label_V[i][0] = 1e10  # first item void = 1e10
        return self.content_label_V

    def random_game(self):
        self.add_vehicle_pos()
        self.update_position()
        self.cal_neighbor()
        # need modification

        self.immediate_reward()
        # generate input state