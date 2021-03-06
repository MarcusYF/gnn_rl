import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from supervised.parse_sample import data_handler
from Analysis.episode_stats import test_summary
from k_cut import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
current_path = os.getcwd()
parent_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
model_names = ['sup_0122_nox_lp', 'sup_0122_nox_l', 'sup_0122_nox_l1', 'sup_0122_nox_l2', 'sup_0122_nox_l3', 'sup_0122_nox_l4', 'sup_0122_nox_l5', 'sup_0122_nox_l6']
for model_name in model_names[:]:

    model_name = 'sup_0122_nox_l'
    version, smooth = 20000, 100

    folder = parent_path + '/Models/' + model_name + '/'

    with open(folder + 'sup_' + str(version), 'rb') as model_file:
        model = pickle.load(model_file)

    # print(model.layers[0].apply_mod.use_x)
    # plot Q-loss/Reward curve
    fig_name = 'loss-curve-' + '-'.join(model_name.split('_')[1:])
    start = smooth
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


    problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16)
    test = test_summary(alg=model, problem=problem, num_instance=1, q_net='mlp')
    test.run_test(episode_len=10, explore_prob=.0, time_aware=False, softmax_constant=50e10)
    test.show_result()

# dh = data_handler(num_chunks=40, batch_size=1000)
# test_set = dh.sample_batch(batch_idx=30000)

Avg value of initial S: 4.732310013771057
Avg max gain: 0.3451266461610794
Avg percentage max gain: 0.06253878
Percentage of instances with positive gain: 0.59

Avg value of initial S: 4.632435812950134
Avg max gain: 0.31661356151103975
Avg percentage max gain: 0.06107664
Percentage of instances with positive gain: 0.58

softmax-100
Avg value of initial S: 4.868634629249573
Avg max gain: 0.6165381020307541
Avg percentage max gain: 0.114936925
Percentage of instances with positive gain: 0.78


base_l (0.1, 1.6)
Avg value of initial S: 4.711408858299255
Avg max gain: 0.8286361070275307
Avg percentage max gain: 0.16415659
Percentage of instances with positive gain: 0.786

nox-l (0.1, 1.6)
Avg value of initial S: 4.684697070360183
Avg max gain: 0.809381370306015
Avg percentage max gain: 0.16202599
Percentage of instances with positive gain: 0.765

nox-l2 (0.1, 0.0)
Avg value of initial S: 4.665292110919952
Avg max gain: 0.74903350263834
Avg percentage max gain: 0.14745723
Percentage of instances with positive gain: 0.755


nox-l3 (0.1, 20.0)
Avg value of initial S: 4.712306947231292
Avg max gain: 0.8442909387350083
Avg percentage max gain: 0.16706638
Percentage of instances with positive gain: 0.804

nox-l4 (0.01, 1.6)
Avg value of initial S: 4.685684136629105
Avg max gain: 0.9405915467143059
Avg percentage max gain: 0.18631606
Percentage of instances with positive gain: 0.823

nox-l5 (0.01, 20)
Avg value of initial S: 4.723482153177262
Avg max gain: 1.166285614013672
Avg percentage max gain: 0.23229663
Percentage of instances with positive gain: 0.897

0122
nox-l
Avg value of initial S: 4.704996926546097
Avg max gain: 0.7067270652651787
Avg percentage max gain: 0.14017354
Percentage of instances with positive gain: 0.753
nox-lp
Avg value of initial S: 4.675292971134186
Avg max gain: 0.7483645051717758
Avg percentage max gain: 0.14385861
Percentage of instances with positive gain: 0.74
nox-l1
Avg value of initial S: 4.709776241779327
Avg max gain: 0.5860170906782151
Avg percentage max gain: 0.10888956
Percentage of instances with positive gain: 0.66


nox-l2
Avg value of initial S: 4.578584158420563
Avg max gain: 0.2509700793027878
Avg percentage max gain: 0.03935092
Percentage of instances with positive gain: 0.38

nox-l3
Avg value of initial S: 4.6948120665550235
Avg max gain: 0.518048666715622
Avg percentage max gain: 0.098522864
Percentage of instances with positive gain: 0.65

nox-l4
Avg value of initial S: 4.716310503482819
Avg max gain: 0.6676579427719116
Avg percentage max gain: 0.12741084
Percentage of instances with positive gain: 0.67

nox-l5
Avg value of initial S: 4.521267337799072
Avg max gain: 0.39519389867782595
Avg percentage max gain: 0.06950233
Percentage of instances with positive gain: 0.6

nox-l6
Avg value of initial S: 4.624367597103119
Avg max gain: 0.8124199438095093
Avg percentage max gain: 0.15596737
Percentage of instances with positive gain: 0.83


One graph test: Rotate state/ 3 loss funcs/look at q-value diff. Refactor the test code.
How can we determine the effectiveness of gnn/q-net?

import timeit
timeit.timeit('generate_G(k=3, m=3, adjacent_reserve=5, hidden_dim=16)'
              , setup='from k_cut import generate_G;'
              , number=100)

timeit.timeit('gg.generate_G()'
              , setup='from k_cut import GraphGenerator; gg = GraphGenerator(3,3,5)'
              , number=100)

g1 = generate_G(k=3, m=3, adjacent_reserve=5, hidden_dim=16)
g1 = g1['g']
x1 = g1.ndata['x']
g.edata['d']

g.edata['d']
g.edata['e_type']


gg = GraphGenerator(3,3,5)
g = gg.generate_G()

problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16)
e0 = dc(problem.g.edata['e_type'])
state, reward = problem.step((0, 2))
e1 = dc(state.edata['e_type'])
state, reward = problem.step((0, 2))
e2 = dc(state.edata['e_type'])
state, reward = problem.step((0, 2), way=1)
e3 = dc(state.edata['e_type'])
state, reward = problem.step((0, 2), way=1)
e4 = dc(state.edata['e_type'])


problem.reset()
timeit.timeit('[gg.generate_G() for i in range(100)]'
              , setup='from k_cut import GraphGenerator; gg = GraphGenerator(3,3,5)'
              , number=100)  # 18s

timeit.timeit('gg.generate_batch_G(batch_size=10)'
              , setup='from k_cut import GraphGenerator; gg = GraphGenerator(3,3,5)'
              , number=100)
# 2.4s

timeit.timeit('gg = [dgl.DGLGraph() for i in range(100)]; [g.add_nodes(9) for g in gg]'
              , setup='import dgl'
              , number=100)


timeit.timeit('problem.reset(compute_S=False)'
              , setup='from k_cut import KCut_DGL; problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16)'
              , number=100)  # 0.2s


timeit.timeit('problem.step((0,2))'
              , setup='from k_cut import KCut_DGL; problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16)'
              , number=100)

G = GraphGenerator(3,4,5)
bg = G.generate_batch_G(batch_size=10)
bg = dgl.unbatch(bg)
bg[7].ndata['x']
bg[7].ndata['label']

g7 = G.generate_G(x=bg[7].ndata['x'], label=[1,2,0,0,1,1,0,2,1,2,0,2])
bg[7].ndata['x'] - g7.ndata['x']
bg[7].ndata['label'] - g7.ndata['label']
bg[7].edata['e_type'] - g7.edata['e_type']
bg[7].edata['d'] - g7.edata['d']
bg[7].edata['w'] - g7.edata['w']


from k_cut import *
from DQN import *
problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16, sample_episode=5)

bg = to_cuda(problem.gen_batch_graph(batch_size=5))
g = [g for g in dgl.unbatch(bg)]

g0 = dc(g)

bg.ndata['label'][:9,:]
g[0].ndata['label']

bg0 = dc(bg)
_, rewards = problem.step_batch(bg, torch.tensor([[0,1],[2,3],[2,4],[1,5],[1,8]]).cuda())
[ 0.0000,  0.0000,  0.3813, -0.1092,  1.8039]
g0_2, r = problem.step(state=g[1], action=(2,3))

def check(g1, g2):
    print(g2.ndata['x'] - g1.ndata['x'])
    print(g2.ndata['label'] - g1.ndata['label'])
    print(g2.edata['e_type'] - g1.edata['e_type'])
    print(g2.edata['d'] - g1.edata['d'])
    print(g2.edata['w'] - g1.edata['w'])

check(g[1], bg1)

problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16, sample_episode=100)
test = test_summary(alg=alg, problem=problem, q_net='mlp')
test.run_test(batch_size=100, gnn_step=3, episode_len=50, explore_prob=0.0)
test.show_result()

a = 0
for i in range(100):
    a += sum(ep[i].reward_seq)