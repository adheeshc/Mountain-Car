import matplotlib.pyplot as plt
import gym              
import numpy as np
np.set_printoptions(suppress=True)
import time

env = gym.make('MountainCar-v0')
env.seed(42);

def truncate(n, decimals=2):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

print(f"State space (gym calls it observation space) is {env.observation_space}")
print(f"\nAction space is {env.action_space}")
print(f'\ncart posiiton is between {truncate(env.observation_space.low[0])} and {truncate(env.observation_space.high[0])}')
print(f'cart velocity is between {truncate(env.observation_space.low[1])} and {truncate(env.observation_space.high[1])}')

# Parameters
NUM_STEPS = 200
NUM_EPISODES = 5000
LEN_EPISODE = 200

# alpha=0.2
# gamma=0.9
# epsilon= 0.8

alpha=0.2
epsilon = .8
gamma = .9
min_eps=0

num_states=(env.observation_space.high-env.observation_space.low)*np.array([10,100])
num_states = np.round(num_states, 0).astype(int) + 1
#print(num_states)
Q=np.zeros((num_states[0],num_states[1],env.action_space.n))
#print(Q.shape)

def discretize_state(state,env_low):
    disc_state=(state - env_low)*np.array([10, 100])
    disc_state = np.round(disc_state, 0).astype(int)
    return disc_state

def plot_reward():
    fig = plt.figure(1)
    plt.clf()
    plt.xlim([0,NUM_EPISODES])
    plt.plot(reward_history,'ro')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Per Episode')
    plt.pause(0.01)
    fig.canvas.draw()

reduction = (epsilon - min_eps)/NUM_EPISODES
reward_history = []
reward_history2 = []
ave_reward_list = []
# Run for NUM_EPISODES
for episode in range(NUM_EPISODES):
    episode_reward = 0
    reward = 0
    curr_state = env.reset()
    done = False
    #for step in range(LEN_EPISODE):
        # Comment to stop rendering the environment
        # If you don't render, you can speed things up
    
    curr_state_disc=discretize_state(curr_state,env.observation_space.low)
    
    while done!=True:
        if episode >= (NUM_EPISODES - 20):
            time.sleep(0.01)
            env.render()
        # Randomly sample an action from the action space
        # Should really be your exploration/exploitation policy
            #action = env.action_space.sample()
        if np.random.random() < 1 - epsilon:
            action = np.argmax(Q[curr_state_disc[0], curr_state_disc[1]]) 
        else:
            action = np.random.randint(0, env.action_space.n)

        # Step forward and receive next state and reward
        # done flag is set when the episode ends: either goal is reached or
        #       200 steps are done
        next_state, reward, done, _ = env.step(action) 
        
        # next_state_disc = (next_state - env.observation_space.low)*np.array([10, 100])
        # next_state_disc = np.round(next_state_disc, 0).astype(int)
        next_state_disc=discretize_state(next_state,env.observation_space.low)
        # This is where your NN/GP code should go
        # Create target vector
        # Train the network/GP
        # Update the policy

        if done and next_state[0] >= 0.5:
            Q[curr_state_disc[0], curr_state_disc[1], action] = reward
                #print(f'finished after {step} timesteps')
            
        else:
            Qchange = alpha*(reward + gamma*np.max(Q[next_state_disc[0],next_state_disc[1]]) - Q[curr_state_disc[0], curr_state_disc[1],action])
            Q[curr_state_disc[0], curr_state_disc[1],action] += Qchange
        
        # Record history
        episode_reward += reward
        
        # Current state for next step
        curr_state_disc = next_state_disc

        #Epsilon Dacey
    if epsilon > min_eps:
        epsilon -= reduction   

         # Record history
    reward_history.append(episode_reward)
    reward_history2.append(episode_reward)
    if (episode+1) % 100 == 0:
        ave_reward = np.mean(reward_history2)
        ave_reward_list.append(ave_reward)
        plot_reward()
        reward_history2 = []
        print(f'Episode {episode+1} Average Reward: {ave_reward}')
        
    
    env.close()