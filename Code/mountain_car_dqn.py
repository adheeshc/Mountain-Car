import matplotlib.pyplot as plt
import gym          # Tested on version gym v. 0.14.0 and python v. 3.17
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import Adam
from keras import backend as K
import random
import pickle as pkl
import time

np.set_printoptions(suppress=True)

class DQNAgent:
  def __init__(self, state_space, action_space):
      self.state_space = state_space
      self.action_space = action_space
      self.alpha=0.002
      self.epsilon=1.0
      self.gamma=0.95
      self.min_eps = 0.01
      self.epsilon_decay = 0.997
      # self.alpha = 0.2
      # self.epsilon = 0.8
      # self.gamma = 0.9  
      # self.min_eps = 0.0
      # self.epsilon_decay = 0.1
      self.storage = []
      self.loss=[]
      self.ep_rewards=[]
      self.model = self.build_model()
      self.target_model = self.build_model()
      self.update_target_model()


  def build_model(self):
      model = Sequential()
      model.add(Dense(64, input_dim=2))
      model.add(Activation('relu'))
      model.add(Dense(64))
      model.add(Activation('relu'))
      model.add(Dense(self.action_space, activation='linear'))
      model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.alpha))
      return model

  def update_target_model(self):
      self.target_model.set_weights(self.model.get_weights())

  def sarsa(self, state, action, reward, next_state, done):
      self.storage.append((state, action, reward, next_state, done))
      self.ep_rewards.append(reward)

  def epsilon_policy(self, state):
      if np.random.rand() <= self.epsilon:
          return random.randrange(self.action_space)
      act_values = self.model.predict(state)
      #print(act_values)
      return np.argmax(act_values[0]) 

  def batch_training(self, batch_size):
  	minibatch = random.sample(self.storage, batch_size)
  	self.loss = []
  	for curr_state, action, reward, next_state, done in minibatch:
  	  Q_update = self.model.predict(curr_state)
  	  if done:
  	  	Q_update[0][action] = reward
  	  else:
  	  	Q_change  = self.target_model.predict(next_state)[0]
  	  	Q_update[0][action] = reward + self.gamma * np.amax(Q_change)
  	  history=self.model.fit(curr_state, Q_update,verbose=0,epochs=1)
  	  self.loss.append(history.history['loss'])
  	  self.ep_rewards=[]
  	mean_loss = np.mean(self.loss)
  	if self.epsilon > self.min_eps:
  		self.epsilon *= self.epsilon_decay
  	return history,mean_loss


  def save(self, name):
      self.model.save(name)

def truncate(n, decimals):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

def plot_loss():
  fig=plt.figure(3)
  plt.clf()
  plt.xlim([0,NUM_EPISODES])
  plt.plot(error)
  plt.xlabel("Episodes")
  plt.ylabel("Average Error")
  plt.title("Average_Loss Vs Episodes")
  plt.pause(0.01)
  fig.canvas.draw()

def plot_epsilon():
  fig=plt.figure(2)
  plt.clf()
  plt.xlim([0,NUM_EPISODES])
  plt.plot(epsilon)
  plt.xlabel("Episode")
  plt.ylabel("Epsilon value")
  plt.title("Epsilon Vs Episodes")
  plt.pause(0.01)  
  fig.canvas.draw()

def plot_reward():
  fig = plt.figure(1)
  plt.clf()
  plt.xlim([0,NUM_EPISODES])
  plt.plot(reward_history,'ro')
  plt.xlabel('Episode')
  plt.ylabel('Reward')
  plt.title(f'Reward Per Episode (NUM_STEPS={NUM_STEPS})')
  plt.pause(0.01)
  fig.canvas.draw()

env = gym.make('MountainCar-v0')
env=env.unwrapped
env.seed(42);

# Print some info about the environment
print("\n========================================================")        
print(f"State space (gym calls it observation space) is {env.observation_space}")
print(f"Action space is {env.action_space}")
print(f'\ncart posiiton is between {truncate(env.observation_space.low[0],2)} and {truncate(env.observation_space.high[0],2)}')
print(f'cart velocity is between {truncate(env.observation_space.low[1],2)} and {truncate(env.observation_space.high[1],2)}')
print("\n========================================================")        

state_space = env.observation_space.shape[0]
action_space = env.action_space.n
# Parameters

NUM_EPISODES = 10
NUM_STEPS = int(NUM_EPISODES/5)
LEN_EPISODE = 200

reward_history = []
reward_history2 = []
ave_reward_list=[]
epsilon=[]
done = False
batch_size = 128
#batch_size = int(LEN_EPISODES)/3
reward = 0
episode_reward=0
agent = DQNAgent(state_space, action_space)
error=[]
#reduction=epsilon-min_eps/NUM_EPISODES
#print(time_now)
# Run for NUM_EPISODES
for episode in range(NUM_EPISODES):
  curr_state = env.reset()
  curr_state = np.reshape(curr_state, [1, state_space])
  flag = 0
  time_now = time.time()
    
  for step in range(LEN_EPISODE):
    # Comment to stop rendering the environment
    # If you don't render, you can speed things up
    if episode >= (NUM_EPISODES - 2):
        #time.sleep(0.01)
        env.render()
    # Randomly sample an action from the action space
    # Should really be your exploration/exploitation policy
        #action = env.action_space.sample()
    action = agent.epsilon_policy(curr_state)

    # Step forward and receive next state and reward
    # done flag is set when the episode ends: either goal is reached or
    #       200 steps are done
    next_state, reward, done, _ = env.step(action) 
    
    # next_state_disc = (next_state - env.observation_space.low)*np.array([10, 100])
    # next_state_disc = np.round(next_state_disc, 0).astype(int)
    if next_state[1] > curr_state[0][1] and next_state[1]>0 and curr_state[0][1]>0:
      reward += 15
    elif next_state[1] < curr_state[0][1] and next_state[1]<=0 and curr_state[0][1]<=0:
      reward +=15
    # This is where your NN/GP code should go
    # Create target vector
    # Train the network/GP
    # Update the policy

    if done:
    	reward = reward + 10000
    else:
      reward=reward-10
    
    next_state = np.reshape(next_state, [1, state_space])
    agent.sarsa(curr_state, action, reward, next_state, done)

    # Record history
    episode_reward += reward
    
    # Current state for next step
    curr_state = next_state

    if done:
      flag=1
      agent.update_target_model()
      break
    if len(agent.storage) > batch_size:
      history,mean_loss=agent.batch_training(batch_size)
      error.append(mean_loss)
  time_later = time.time()
       # Record history
  reward_history.append(-episode_reward)
  reward_history2.append(-episode_reward)
  epsilon.append(agent.epsilon)

  # if flag==0:
  #   print(f'\nCOMPLETED SUCCESSFULLY IN {episode} episodes')
  #   print(f'NOW OPTIMIZING\n')
  if (episode+1) % NUM_STEPS == 0:
      ave_reward = np.mean(reward_history2)
      ave_reward_list.append(ave_reward)
      plot_reward()
      #plot_epsilon()
      #plot_loss()
      reward_history2 = []
      print(f'Episode {episode+1} Average Reward: {ave_reward}')       
  if episode==NUM_EPISODES-1:
  	agent.save(f"{NUM_EPISODES}it_{batch_size}batch.h5")
  env.close()
