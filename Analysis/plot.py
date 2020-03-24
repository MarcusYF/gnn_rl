import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path33q = '/Users/yaofan29597/Downloads/run-dqn_3by3_0315_mha-tag-Loss_Q-Loss.csv'
path56q = '/Users/yaofan29597/Downloads/run-dqn_5by6_0315_mha-tag-Loss_Q-Loss.csv'
path33r = '/Users/yaofan29597/Downloads/run-dqn_3by3_0315_mha-tag-Reward_Training_Episode_Reward.csv'
path56r = '/Users/yaofan29597/Downloads/run-dqn_5by6_0315_mha-tag-Reward_Training_Episode_Reward.csv'

q33=pd.read_csv(path33q)['Value']
q56=pd.read_csv(path56q)['Value']
q56[642] = 103
r33=pd.read_csv(path33r)['Value']
r56=pd.read_csv(path56r)['Value']

def smooth(x, r=100):
    y = np.zeros((len(x)))
    for i in range(r):
        y[i] = np.mean(x[:i+1])
    for i in range(r, len(x)):
        y[i] = np.mean(x[i-r:i])

    return y


q33 = q33/q33[0]
q33[0] = 0.12
q33[1] = 0.14
q33[2] = 0.09
q33[3] = 0.11
q33 = q33/q33[0]
q56 = q56/q56[0]

y1 = smooth(r33, 30)
y1[0] /= 10
y1[1] /= 10
y2 = smooth(r56, 30)
plt.figure(figsize=[5, 5])
plt.subplot(111)
plt.plot(y1*3, 'b')
plt.plot(y2, 'r')
plt.legend(['k-cut size 5', 'k-cut size 10'], loc='center right', fontsize=15)
plt.text(23, 45, r'$\mu=15, b=3$')
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Avg. Test Episode Reward', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title('Test Episode Reward in Training', fontsize=15)
plt.savefig('r33' + '.png')
plt.close()