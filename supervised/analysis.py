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
model_names = ['sup_0122_nox_lp', 'sup_0122_nox_l1', 'sup_0122_nox_l2', 'sup_0122_nox_l3', 'sup_0122_nox_l4', 'sup_0122_nox_l5', 'sup_0122_nox_l6']
for model_name in model_names:
    version, smooth = 30000, 100

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
    test = test_summary(alg=model, problem=problem, num_instance=1000)
    test.run_test(episode_len=50, explore_prob=.0, time_aware=False, softmax_constant=50e10)
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

nox-l
Avg value of initial S: 4.704996926546097
Avg max gain: 0.7067270652651787
Avg percentage max gain: 0.14017354
Percentage of instances with positive gain: 0.753