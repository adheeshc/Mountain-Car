import gym
from keras.models import load_model
import numpy as np

model = load_model('mountain_car.h5')
env = gym.make('MountainCar-v0')

NUM_EPISODES = 1

for episode in range(NUM_EPISODES):
    state = env.reset()
    done = False
    time_taken = 0
    while not done:
        env.render()
        time_taken +=1
        state = np.reshape(state, [1,2])
        action = model.predict(state)
        action = np.argmax(action)
        next_state, reward, done, _ = env.step(action)
        state = next_state
    print('time taken:', time_taken)