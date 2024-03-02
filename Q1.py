
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cdist
import sys

dim = [1, 2, 4, 8, 16, 32, 64]
sample_size = 10**6

data_points = [np.random.uniform(0, 1, (sample_size, d)).round(5) for d in dim]

l_norm = []

for i in range(len(dim)):
    r_l1 = []
    r_l2 = []
    r_linf = []
    temp = []

    for j in range(100):
        dist_l1 = cdist(data_points[i], [data_points[i][j]], metric='cityblock')
        dist_l2 = cdist(data_points[i], [data_points[i][j]], metric='euclidean')
        dist_linf = cdist(data_points[i], [data_points[i][j]], metric='chebyshev')

        r_l1 = np.max(dist_l1) / np.min(dist_l1[dist_l1 != 0])
        r_l2 = np.max(dist_l2) / np.min(dist_l2[dist_l2 != 0])
        r_inf = np.max(dist_linf) / np.min(dist_linf[dist_linf != 0])
        temp.append([r_l1, r_l2, r_inf])

    l_norm.append(np.mean(np.transpose(temp), axis=1))

x=np.transpose(l_norm)
dim= [str(x) for x in dim]
#print(l_norm)


plt.plot(dim[:],x[:][0], label='L1 norm')
plt.plot(dim[:],x[:][1], label='L2 norm')
plt.plot(dim[:],x[:][2], label='Linf norm')
plt.xticks(dim)
plt.xlabel('Dimension')
plt.ylabel('Average Ratio')
plt.legend()
plt.savefig("p1.png")
plt.show()

# x=np.log(x)

# plt.plot(dim[:],x[:][0], label='L1 norm')
# plt.plot(dim[:],x[:][1], label='L2 norm')
# plt.plot(dim[:],x[:][2], label='Linf norm')
# plt.xticks(dim)
# plt.xlabel('Dimension')
# plt.ylabel('Logarithmic Average Ratio')
# plt.legend()
# plt.savefig("p2.png")
# plt.show()
