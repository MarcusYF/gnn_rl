import os
import pickle
import time
from k_cut import *
from DQN import *
import torch
from tqdm import tqdm
import numpy as np
from Analysis.episode_stats import test_summary
import random
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from Qiter_swap import state2QtableKey, QtableKey2state, gen_comb012




all_state = set([state2QtableKey([int(i) for i in s[:-1].split(',')], reduce_rotate=True) for s in gen_comb012(3, 3, 3)])


state_visit_count = {}.fromkeys(all_state, 0)
begin_state = QtableKey2state('0,1,2,2,1,1,2,0,0')
for k in tqdm(range(1000000)):
    i, j = random.sample([0,1,2,3,4,5,6,7,8], 2)
    state_ = dc(begin_state)
    begin_state[i], begin_state[j] = begin_state[j], begin_state[i]
    new_ = state2QtableKey(begin_state)
    old_ = state2QtableKey(state_)
    if new_ == old_:
        continue
    else:
        state_visit_count[new_] += 1
