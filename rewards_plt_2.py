# a 2d plot

import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools

# infile = open('Helsinki_TD3_GroupMask_site04[0][A]_2021-05-08.pickle','rb')
# new_dict = pickle.load(infile)
# infile.close()

with open('Helsinki_TD3_GroupMask_site04[0][A]_2021-05-08.pickle', 'rb') as file:
#with open('Helsinki_BCQ_25_05_Test4.pickle', 'rb') as file:
    new_dict = pickle.load(file)




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

#uncomment when you want to generate a buffer to train BCQ:
# buffer_name = 'Helsinki_TD3_GroupMask_site04[0][A]_2021-05-08'
# np.save(f"./buffers/{buffer_name}__0_state.npy", state_batch)
# np.save(f"./buffers/{buffer_name}__0_action.npy", action_batch)
# np.save(f"./buffers/{buffer_name}__0_reward.npy", reward_batch)
# np.save(f"./buffers/{buffer_name}__0_next_stat", next_state_batch)
plt.plot(moving_average(reward_batch,n=10))
plt.show()