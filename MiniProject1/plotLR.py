import pip
from pip._internal import main
import os
#os.system('/bin/bash -c "sudo pip install numpy matplotlib scipy pandas"')
import numpy as np
import math
import scipy
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import time
from classification import logisticRegression
from classification import LDA
from tempfile import TemporaryFile
'''
learningRate = np.arange(0.01,0.3,0.01)
result = np.load('resultWine2.npy')

fig,ax1 = plt.subplots()
ax1.set_title("Training Time, Accuracy vs. Initial Learning Rate")
color = 'tab:red'
ax1.set_xlabel('Initial Learning Rate')
ax1.set_ylabel('Training Time (s)', color=color)
ttline, = ax1.plot(learningRate,result[:,1], color = color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Accuracy(%)', color=color)  # we already handled the x-label with ax1
accline, = ax2.plot(learningRate,result[:,0],color = color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
lg = ax1.legend([ttline,accline],['Training Time(s)','Accuracy(%)'],loc='lower center')
plt.show()


result = np.load('resultBC2.npy')
fig,ax1 = plt.subplots()
ax1.set_title("Training Time, Accuracy vs. Initial Learning Rate")
color = 'tab:red'
ax1.set_xlabel('Initial Learning Rate')
ax1.set_ylabel('Training Time (s)', color=color)
ttline, = ax1.plot(learningRate,result[:,1], color = color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Accuracy(%)', color=color)  # we already handled the x-label with ax1
accline, = ax2.plot(learningRate,result[:,0],color = color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
lg = ax1.legend([ttline,accline],['Training Time(s)','Accuracy(%)'],loc='lower center')
plt.show()




power = np.arange(1,4,0.5)
result = np.load('resultWinePower.npy')
fig,ax1 = plt.subplots()
ax1.set_title("Training Time, Accuracy vs. Learning Rate Decay Speed")
color = 'tab:red'
ax1.set_xlabel('Learning Rate Decay Speed')
ax1.set_ylabel('Training Time (s)', color=color)
ttline, = ax1.plot(power,result[:,1], color = color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Accuracy(%)', color=color)  # we already handled the x-label with ax1
accline, = ax2.plot(power,result[:,0],color = color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
lg = ax1.legend([ttline,accline],['Training Time(s)','Accuracy(%)'],loc='lower center')
plt.show()


result = np.load('resultBCPower.npy')
fig,ax1 = plt.subplots()
ax1.set_title("Training Time, Accuracy vs. Learning Rate Decay Speed")
color = 'tab:red'
ax1.set_xlabel('Learning Rate Decay Speed')
ax1.set_ylabel('Training Time (s)', color=color)
ttline, = ax1.plot(power,result[:,1], color = color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Accuracy(%)', color=color)  # we already handled the x-label with ax1
accline, = ax2.plot(power,result[:,0],color = color)
ax2.tick_params(axis='y', labelcolor=color)
# ax2.set_ylim([98.25,98.75])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
lg = ax1.legend([ttline,accline],['Training Time(s)','Accuracy(%)'],loc='lower center')
plt.show()


'''
colToDelete = np.arang(1,12,1)
result = np.load('resultWineFeatures.npy')

fig,ax1 = plt.subplots()
ax1.set_title("Training Time, Accuracy vs. Initial Learning Rate")
color = 'tab:red'
ax1.set_xlabel('Feature')
ax1.set_ylabel('Training Time (s)', color=color)
ttline, = ax1.plot(colToDelete,result[:,1], color = color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Accuracy(%)', color=color)  # we already handled the x-label with ax1
accline, = ax2.plot(colToDelete,result[:,0],color = color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# lg = ax1.legend([ttline,accline],['Training Time(s)','Accuracy(%)'],loc='lower center')
plt.show()
