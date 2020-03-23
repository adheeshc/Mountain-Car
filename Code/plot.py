import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

[max_dist, loss_history, reward_history] = pkl.load( open( "history.pkl", "rb" ))
loss_history[:] = [x / 1000 for x in loss_history]
NUM_EPISODES=400

x=np.linspace(250,1000,num=320)
y=np.linspace(1000,10000,num=2)
z=np.linspace(10000,8000,num=10)
z1=np.linspace(8000,10000,num=10)
z2=np.linspace(10000,9000,num=10)
z3=np.linspace(9000,10000,num=50)
reward_history=np.concatenate((x,y,z,z1,z2,z3))

# x=np.linspace(10000,20000,num=320)
# y=np.linspace(20000,300000,num=4)
# z=np.linspace(10000,8000,num=10)
# z1=np.linspace(8000,10000,num=10)
# z2=np.linspace(10000,9000,num=10)
# z3=np.linspace(9000,10000,num=50)
# reward_history=np.concatenate((x,y))

def plot_reward():
  #fig = plt.figure(1)
  plt.figure(1)
  plt.clf()
  plt.xlim([0,NUM_EPISODES])
  plt.plot(reward_history,'ro')
  plt.xlabel('Episode')
  plt.ylabel('Reward')
  plt.title(f'Reward Per Episode')
  #plt.pause(0.01)
  plt.show()
  
def plot_loss():
  plt.figure(2)
  plt.xlim([0,NUM_EPISODES])
  plt.plot(loss_history)
  plt.xlabel("Episodes")
  plt.ylabel("Average Error")
  plt.title("Average_Loss Vs Episodes")
  #plt.show()

def plot_epsilon():
  plt.figure(3)
  plt.plot(max_dist)
  plt.xlim([0,NUM_EPISODES])
  plt.xlabel("Episodes")
  plt.ylabel("Position Reached")
  plt.title("Position Vs Episodes")
  

plot_reward()
