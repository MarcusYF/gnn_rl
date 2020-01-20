import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from supervised.parse_sample import data_handler
from Analysis.episode_stats import test_summary
from k_cut import *

current_path = os.getcwd()
parent_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
folder = parent_path + '/Models/sup_0119_base/'
version, smooth = 20000, 100
with open(folder + 'sup_' + str(version), 'rb') as model_file:
    model = pickle.load(model_file)


# plot Q-loss/Reward curve
fig_name = 'loss-curve-0119-base'
start = 100
x1 = range(1+start, version)
y1 = model.h_residual[start:]

x2 = range(smooth+start, version)
y2 = [np.mean(model.h_residual[i:i+smooth]) for i in range(start, version-smooth)]


fig = plt.figure(figsize=[5, 5])
ax = fig.add_subplot(111)
ax.plot(x1, y1, label='batch loss')
ax.plot(x2, y2, label='smooth batch loss', color='r')
ax.set_xlabel('Training Epochs')
ax.set_ylabel("Quadratic Loss")
ax.set_title('Training Loss Curve')
plt.savefig(current_path + '/supervised/figs/' + fig_name + '.png')
plt.close()

folder = parent_path + '/Models/dqn_0114_base/'
with open(folder + 'dqn_' + str(5500), 'rb') as model_file:
    model = pickle.load(model_file)
problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16)
test = test_summary(alg=model, problem=problem, num_instance=100)
test.run_test(episode_len=50, explore_prob=.0, time_aware=False)
test.show_result()

dh = data_handler(num_chunks=40, batch_size=1000)
test_set = dh.sample_batch(batch_idx=30000)


Avg value of initial S: 4.574587857723236
Avg max gain: 0.8197347176074982
Avg max gain budget: 50.0
Var max gain budget: 0.0
Avg percentage max gain: 0.16990764
Percentage of instances with positive gain: 0.78