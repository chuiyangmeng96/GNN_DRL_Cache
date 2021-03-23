import numpy as np
import pandas as pd

data = np.array(pd.read_csv("koln-pruned45.000000.csv"))
print(np.median(data[:, 2]))
print(np.median(data[:, 3]))


################
# 1st thing is whether we should centralize vehicle area to urban (10*10km)
################


# select adj matrix
# vehicle id should be stationary which means discarded or newly-enrolled vehicles should be
# taken into consideration prior to adj matrix setup
# for item in ls:
#     item = np.array(item)
#     dim = item.shape[0]
#     for i in range(dim):









