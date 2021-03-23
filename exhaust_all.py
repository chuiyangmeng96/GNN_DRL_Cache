import numpy as np
import math
import matplotlib.pyplot as plt
from envcopy import Environ
import sys

env = Environ() # NEED MODIFICATION: combine env with other functions
time_delta = 1.5
vehicle_num = 500
RSU_num = 4
content_num = 50
content_size = 200
cache_num_V = 3
cache_num_R = 10
RSUposition = np.array([[250, 0], [-250, 0], [0, -250], [0, 250]])

band_V2V = 5e6
p_k = pow(10, 24 / 10)
h_ik = 0.16
alpha = 2
sigma_2 = pow(10, -11)

bij = 1e7
p_j = pow(10, 36 / 10)
h_ij = 0.16

obs_num = 100
eps = pow(10, -8)


def EXHAUSTALL(position, candidateV, candidateR, content_label_V, content_label_R, rand_content, reward, time_delta):
    packet_loss = np.zeros(vehicle_num)
    T_delay = np.zeros(vehicle_num)
    count = 0
    for i in range(vehicle_num):
        indexV = np.nonzero(candidateV[i, :])[0] # return index of nonzero entries
        indexR = np.nonzero(candidateR[i, :])[0]
        rowV = []
        rowR = []
        for numberV in indexV:
            rowV.append([i, numberV, candidateV[i, numberV]]) # fill in nonzero entries
        for numberR in indexR:
            rowR.append([i, numberR, candidateR[i, numberR]])
        rowR = np.array(rowR)
        rowV = np.array(rowV)
        # rowR = rowR[rowR[:, 2].argsort()[::-1]]
        # rowV = rowV[rowV[:, 2].argsort()[::-1]]
        rowR = np.flipud(rowR[rowR[:, 2].argsort()])
        rowV = np.flipud(rowV[rowV[:, 2].argsort()])
        disV = 1e10
        disR = 1e10
        tempV = None
        tempR = None
        for num in range(rowV.shape[0]):
            tmp = int(rowV[num][1])
            if rand_content[i] in content_label_V[tmp, :]:
                if rowV[num][2] < disV:
                    disV = rowV[num][2]
                    tempV = rowV[num][1]

        for num in range(rowR.shape[0]):
            tmp = int(rowR[num][1])
            if rand_content[i] in content_label_R[tmp, :]:
                if rowR[num][2] < disR:
                    disR = rowR[num][2]
                    tempR = rowR[num][1]

        if tempV is not None and tempR is not None:
            if disV < disR:  # request from vehicle
                targetV = int(tempV)  # find nearest vehicle index
                if rand_content[i] in content_label_V[targetV, :]:  # content in nearest vehicle
                    d_ik = np.sqrt(np.sum(pow((position[targetV, 0:2] - position[i, 0:2]), 2)))
                    R_ki = band_V2V * (np.log2(sigma_2 + p_k * h_ik * pow(d_ik + eps, -alpha)) - np.log2(sigma_2)) / 1e6
                    T_if = content_size / (R_ki + eps)
                    T_delay[i] = T_if
                    if T_if > time_delta:
                        packet_loss[i] = 1
                        reward += -10
                    else:
                        reward += 100 / T_if  # compute latency and reward
                else:  # content not in vehicle, then receive from MBS
                    T_delay[i] = 1e10
                    reward += -10  # penalty
                    count += 1
            else:  # request from RSU
                targetR = int(tempR)  # find nearest RSU index
                if rand_content[i] in content_label_R[targetR, :]:  # content in nearest RSU
                    d_ij = np.sqrt(np.sum(pow((RSUposition[targetR, 0:2] - position[i, 0:2]), 2)))
                    R_ji = bij * (np.log2(sigma_2 + p_j * h_ij * pow(d_ij + eps, -alpha)) - np.log(sigma_2)) / 1e6
                    T_if = content_size / (R_ji + eps)
                    T_delay[i] = T_if
                    if T_if > time_delta:
                        packet_loss[i] = 1
                        reward += -10
                    else:
                        reward += 100 / T_if  # compute latency and reward
                else:  # content not in RSU, then receive from MBS
                    T_delay[i] = 1e10
                    reward += -10  # penalty
                    count += 1
        elif tempV is not None and tempR is None:
            targetV = int(tempV)  # find nearest vehicle index
            if rand_content[i] in content_label_V[targetV, :]:  # content in nearest vehicle
                d_ik = np.sqrt(np.sum(pow((position[targetV, 0:2] - position[i, 0:2]), 2)))
                R_ki = band_V2V * (np.log2(sigma_2 + p_k * h_ik * pow(d_ik + eps, -alpha)) - np.log2(sigma_2)) / 1e6
                T_if = content_size / (R_ki + eps)
                T_delay[i] = T_if
                if T_if > time_delta:
                    packet_loss[i] = 1
                    reward += -10
                else:
                    reward += 100 / T_if  # compute latency and reward
            else:  # content not in vehicle, then receive from MBS
                T_delay[i] = 1e10
                reward += -10  # penalty
                count += 1
        elif tempV is None and tempR is not None:
            targetR = int(tempR)  # find nearest RSU index
            if rand_content[i] in content_label_R[targetR, :]:  # content in nearest RSU
                d_ij = np.sqrt(np.sum(pow((RSUposition[targetR, 0:2] - position[i, 0:2]), 2)))
                R_ji = bij * (np.log2(sigma_2 + p_j * h_ij * pow(d_ij + eps, -alpha)) - np.log(sigma_2)) / 1e6
                T_if = content_size / (R_ji + eps)
                T_delay[i] = T_if
                if T_if > time_delta:
                    packet_loss[i] = 1
                    reward += -10
                else:
                    reward += 100 / T_if  # compute latency and reward
            else:  # content not in RSU, then receive from MBS
                T_delay[i] = 1e10
                reward += -10  # penalty
                count += 1
        else:
            T_delay[i] = 1e10
            reward += -10
            count += 1
    return reward, packet_loss, T_delay, count


# env.random_game()  # initialize state
env.add_vehicle_pos()
content_label_V = np.zeros((vehicle_num, 3))
content_label_R = np.zeros((RSU_num, 10))
total_reward = 0
BS_count = np.zeros(obs_num)
packet_loss = np.zeros(obs_num)
result_list = []
for obs_step in range(obs_num):
    # env.content_request()  # return request_arr which denotes to content request probability
    position = env.update_position(obs_step)   # update position
    request_arr_V, request_arr_R = env.content_request()
    candidateV, candidateR = env.cal_neighbor()

    content_R = np.zeros(RSU_num)
    for i in range(RSU_num):  # randomly request a content for each RSU
        dice = np.random.rand(1)
        if 0 < dice < request_arr_R[i][0]:  # randomly request a content and its label
            content_R[i] = 0
        else:
            for j in range(1, content_num):
                if np.sum(request_arr_R[i, 0:j - 1]) < dice < np.sum(request_arr_R[i, 0:j]):
                    content_R[i] = j

    content_label = np.zeros(vehicle_num)
    for i in range(vehicle_num):  # randomly request a content for each vehicle
        dice = np.random.rand(1)
        if 0 < dice < request_arr_V[i][0]:  # randomly request a content and its label
            content_label[i] = 0
        else:
            count = 0
            for j in range(1, content_num):
                if np.sum(request_arr_V[i, 0:j-1]) < dice < np.sum(request_arr_V[i, 0:j]):
                    content_label[i] = j
    rand_content = content_label

    total_reward, packetloss, T_delay, count = EXHAUSTALL(position, candidateV, candidateR, content_label_V, content_label_R, rand_content, total_reward, time_delta)
    packet_loss[obs_step] = np.count_nonzero(packetloss)
    BS_count[obs_step] = count

    for i in range(vehicle_num):
        if T_delay[i] < time_delta:
            if content_label[i] in content_label_V[i, :]:  # case 1 requested content already in vehicle cache
                content_label_index = list(content_label_V[i, :]).index(content_label[i])
                if content_label_index != 0:
                    for j in range(content_label_index):
                        content_label_V[i][j + 1] = content_label_V[i][j]
                    content_label_V[i][0] = content_label[i]
            else:  # requested content not in vehicle, enough/not enough (same case) space for storing new content
                for j in range(cache_num_V - 1):
                    content_label_V[i][j +1] = content_label_V[i][j]
                content_label_V[i][0] = content_label[i]
        else:
            for j in range(cache_num_V - 1):
                content_label_V[i][j + 1] = content_label_V[i][j]
            content_label_V[i][0] = 1e10  # first item void = 1e10

    for i in range(RSU_num):
        if content_R[i] in content_label_R[i, :]:  # case 1 requested content already in RSU cache
            content_label_index = list(content_label_R[i, :]).index(content_R[i])
            if content_label_index != 0:
                for j in range(content_label_index):
                    content_label_R[i][j + 1] = content_label_R[i][j]
                content_label_R[i][0] = content_R[i]
        else:  # requested content not in RSU
            for j in range(cache_num_R - 1):
                content_label_R[i][j + 1] = content_label_R[i][j]
            content_label_R[i][0] = content_R[i]
    print('\rObservation Step: {}\t\tCumulative Reward: {:.6f}\t\tPacket Loss Rate Per Step: {:.4f}\t\tRate of Content '
          'Received from BS: {:.6f}'.format(obs_step, total_reward, packet_loss[obs_step]/vehicle_num, BS_count[obs_step]/vehicle_num))
    result_list.append(total_reward)

packet_loss_rate_average = np.sum(packet_loss)/(obs_num * vehicle_num)
packet_loss_rate_step = packet_loss/vehicle_num

BS_content_rate_average = np.sum(BS_count)/(obs_num * vehicle_num)
BS_content_rate_step = BS_count/vehicle_num

print('\rTotal Reward: {:.6f}'.format(total_reward))
print('\rPacket Loss Rate: {:.6f}'.format(packet_loss_rate_average))
print('\rAverage Rate of Content Received from BS: {:.6f}'.format(BS_content_rate_average))
obs = list(range(obs_num))

plt.figure(1)
plt.plot(obs, result_list)
plt.xlabel('Time Step (s)')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward vs Time Step in EXHAUST_ALL method')

plt.figure(2)
plt.plot(obs, packet_loss_rate_step)
plt.xlabel('Time Step (s)')
plt.ylabel('Packet Loss Rate')
plt.title('Packet Loss Rate vs Time Step in EXHAUST_ALL method')

plt.figure(3)
plt.plot(obs, BS_content_rate_step)
plt.xlabel('Time Step (s)')
plt.ylabel('Rate of Content Received from BS')
plt.title('Rate of Content Received from BS vs Time Step in EXHAUST_ALL method')
plt.show()
