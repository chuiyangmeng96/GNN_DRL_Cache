import numpy as np
import pandas as pd
import pickle

max_connection = 100 # self definition
data = np.array(pd.read_csv("koln-pruned25.000000.csv"))
center_x = np.median(data[:, 2])
center_y = np.median(data[:, 3])
center = np.array([center_x, center_y])
print(center)
length = data.shape[0]
lower_bound = np.max(np.argwhere(data[:, 0] == data[0][0])) # data lower_bound index
upper_bound = np.min(np.argwhere(data[:, 0] == data[length - 1][0])) # data upper_bound index

ls = []
time_begin = int(data[lower_bound + 1][0])
time_end = int(data[upper_bound][0])

# 1st thing is whether we should centralize vehicle area to urban (r = 3km)
scope_radius = 3e3
for i in range(time_begin, time_end):
    idx = np.argwhere(data[lower_bound + 1:upper_bound, 0].astype(int) == i)
    print(idx[0])
    delete_idx = []
    for j in idx:
        if np.sqrt(np.sum(np.power(data[j, 2:4] - center, 2))) > scope_radius:
            delete_idx.append(j)
    idx = np.delete(idx, delete_idx, axis=0)
    print(delete_idx[0])
    ls.append(data[idx, :])

# select adj matrix
# vehicle id should be stationary which means discarded or newly-enrolled vehicles should be
# taken into consideration prior to adj matrix setup
neighbor_ls = []
for item in ls:
    item = np.array(item)
    dim = item.shape[0]
    # x_outrange = np.argwhere(item[:, 2] > central_x + half_range or item[:, 2] < central_x - half_range)
    # y_outrange = np.argwhere(item[:, 3] > central_y + half_range or item[:, 3] < central_y - half_range)
    # outrange = np.concatenate((x_outrange, y_outrange), 0).reshape(x_outrange.shape[0] + y_outrange.shape[0])
    # _, i = np.unique(outrange, return_index=True)
    # outrange = outrange[np.sort(i)]
    # item = np.delete(item, outrange, axix=0) # delete vehicles out of range

    for i in range(dim):
        for j in range(dim):
            neighbor_matrix = np.zeros((dim, dim))
            dis = np.sqrt(np.sum(np.power(item[i, 2:4] - item[j, 2:4], 2)))
            if i != j and dis < max_connection:
                neighbor_matrix[i][j] = dis
    neighbor_ls.append(neighbor_matrix)
# contain a list of several neighboring matrices with each entry denoting to distance
# each of which can be fed into adj function to calculate adj matrix


