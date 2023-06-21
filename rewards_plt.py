# a 2d plot

import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools

nbr_cells = ['site03[0][A]', 'site03[2][A]', 'site04[1][A]', 'site04[2][A]', 'site08[2][A]']
lho_per_t = []
pho_per_t = []
wcho_per_t = []
eho_per_t = []
alpha = 0.5

infile = open('Helsinki_TD3_GroupMask_site04[0][A]_2021-05-08.pickle','rb')
new_dict = pickle.load(infile)
infile.close()

lho = np.array(list(new_dict[1][i]["late_handovers_count"] for i in range(len(new_dict[1]))))
pho = np.array(list(new_dict[1][i]["pingpong_handovers_count"] for i in range(len(new_dict[1]))))
wcho = np.array(list(new_dict[1][i]["wrong_cell_handovers_count"] for i in range(len(new_dict[1]))))
eho = np.array(list(new_dict[1][i]["early_handovers_count"] for i in range(len(new_dict[1]))))
lho_cumul = np.sum(lho,1)
pho_cumul = np.sum(pho,1)
wcho_cumul = np.sum(wcho,1)
eho_cumul = np.sum(eho,1)
pho_per_t.append(pho_cumul[0])
wcho_per_t.append(wcho_cumul[0])
eho_per_t.append(eho_cumul[0])
lho_per_t.append(lho_cumul[0])

for i in range(1, len(lho_cumul)):
    lho_per_t.append(lho_cumul[i] - lho_cumul[i - 1])
    pho_per_t.append(pho_cumul[i] - pho_cumul[i - 1])
    eho_per_t.append(eho_cumul[i] - eho_cumul[i - 1])
    wcho_per_t.append(wcho_cumul[i] - wcho_cumul[i - 1])

reward = np.array(lho_per_t) + alpha * np.array(pho_per_t) + np.array(eho_per_t) + np.array(wcho_per_t)
plt.plot(reward)
plt.ylabel('Obtained reward')
plt.show()

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# TTT_pool = [0.128, 0.512, 1.024, 2.56]  # in sec
# CIO_pool = [int(x) for x in [-5, -2, 0, 2, 5]]  # in dB
# ls_config = [TTT_pool] + [CIO_pool] * len(nbr_cells)
# permutation_pool = list(itertools.product(*ls_config))
# action = new_dict[2][0][1]
# action_value = permutation_pool[action]

state_batch = [new_dict[2][x][0] for x in range(len(new_dict[2]))]
action_batch = [new_dict[2][x][1] for x in range(len(new_dict[2]))]
reward_batch = [new_dict[2][x][2] for x in range(len(new_dict[2]))]
next_state_batch = [new_dict[2][x][3] for x in range(len(new_dict[2]))]


plt.plot(moving_average(reward_batch,n=50))
plt.show()