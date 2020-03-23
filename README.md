# Mountain-Car
Q Learning and DQN for mountain car

=================================
FILE DESCRIPTIONS
=================================

1)bipedal.py - contains code for BipedalWalker-v2 using NN
2)bipedal_test.py - contains model tester code for BipedalWalker-v2
3)BipedalWalker_model2.h5 - contains trained model file for BipedalWalker-v2

4)mountain_car_dqn.py - contains code for mountainCar-v0 using NN
5)mountain_car_q.py - contains code for mountainCar-v0 using Q Table
6)mountain_car.h5 - contains trained model file for mountainCar-v0
7)mountain_car_test.py - contains tester code for mountainCar-v0 

8)Report.pdf - Report File
9)mean_reward.npy - contains output of mean_rewards obtained for every 100 episodes from BipedalWalker-v2 (open using numpy)
10)reward_history.npy - contains all rewards per episode from BipedalWalker-v2 (open using numpy)


=================================
Libraries Used
=================================

1) Gym
2) Numpy 
3) matplotlib
4) Keras
5) Pickle
6) random
7) time

=================================
RUN INSTRUCTION
=================================

1) Make sure directory structure is maintained 
2) Make sure all the libraries are installed
3) RUN mountain_car_q.py for MountainCar-v0 using Q Table (OPTIONAL)
4) RUN mountain_car_dqn/final_dqn.py for MountainCar-v0 using DQN (PROBLEM 1)
5) RUN mountain_car_test.py for implementation of dqn trained MountainCar-v0 model (for 400 iterations, using 128 batch size)
6) RUN bipedal.py for BipedalWalker-v2 using DQN (PROBLEM 2)
7) RUN bipedal_test.py for implementation of trained BipedalWalker-v2 model (for 1000 iterations, using 16 batch size)
