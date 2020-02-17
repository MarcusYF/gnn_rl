from k_cut import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import os
from tqdm import tqdm
from Analysis.episode_stats import test_summary
from torch.nn import DataParallel
from toy_models.Qiter import vis_g

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

model_name = 'dqn_5by6_0214_in_base'
# model_name = 'dqn_5by6_0213_base'
model_version = str(50000)
k = 5
m = 6
ajr = 10
h = 32
mode = 'incomplete'
q_net = 'mlp'
batch_size = 10
trial_num = 5
sample_episode = batch_size * trial_num
gnn_step = 3
episode_len = 50
explore_prob = 0.0
Temperature = 0.05


folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + model_name + '/'
with open(folder + 'dqn_' + model_version, 'rb') as model_file:
    alg = pickle.load(model_file)

# test summary
problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=h, mode=mode, sample_episode=sample_episode)
test = test_summary(alg=alg, problem=problem, q_net=q_net)
test.run_test(trial_num=trial_num, batch_size=batch_size, gnn_step=gnn_step, episode_len=episode_len, explore_prob=explore_prob, Temperature=Temperature)
test.show_result()

# visualization
g_i = 1
init_state = test.episodes[g_i].init_state
label_history = init_state.ndata['label'][test.episodes[g_i].label_perm]

problem.g = dc(init_state)
problem.reset_label(label=label_history[0].argmax(dim=1))
problem.calc_S()
problem.reset_label(label=label_history[50].argmax(dim=1))
problem.calc_S()

path = os.path.abspath(os.path.join(os.getcwd())) + '/Analysis/eval_model/test1'
vis_g(problem, name=path, topo='cut')

actions = problem.get_legal_actions(state=problem.g)
bg = dgl.batch([problem.g] * actions.shape[0])
problem.sample_episode = actions.shape[0]
problem.gen_step_batch_mask()
problem.step_batch(states=bg, action=actions)

