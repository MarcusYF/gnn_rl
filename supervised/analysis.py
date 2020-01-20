import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

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
