'''
Task: to navigate a 40x40 square grid from 'S' to 'G' using Q learning
Assumptions:
1. The agent can move in 4 directions - up, down, left, right
2. The agent can move only to the neighbouring grids

Hyperparameters:
1. Reward = -1 for each step to find shortest path
2. Value of goal state = 100
3. Value of obstacle = -100

5. epsiolon
6. gamma
7. alpha

For creating the Q table we will use epsilon greedy policy
'''

import numpy as np
import matplotlib.pyplot as plt
import random


with open(r"C:\Users\ganga\Documents\IISc Coursework\2nd Sem\Motion Planning\grid\MotionPlanning_2025\grid.txt") as f:
    grid = f.read().splitlines()

#Convert '.' to 0, 'X' to 1, 'S' to 0, 'G' to 100
grid_dict = {'.': 0, 'X': -1, 'S': 2, 'G': 1}
grid = np.array([[grid_dict[j] for j in i.split()] for i in grid])

#plot the grid
fig, ax = plt.subplots()
img = ax.imshow(grid, cmap='viridis')
plt.show()

start = np.where(grid == 2)
goal = np.where(grid == 1)
start = np.array([start[0][0], start[1][0]])  # Convert to (row, col)
goal = list(zip(goal[0], goal[1]))  # List of all goal coordinates

#Initialize a 40 x 40 x4 grid Q values: each grid has 4 possible actions
Q = np.zeros((40,40,4))
ep = 0.1 #Epsilon for epsilon greedy policy
gamma = 0.9 #Discount factor
alpha = 0.1 #Learning rate
num_episodes = 5000

#Q for all actions at goal state = 100
Q[goal[0], goal[1], :] = 100
#Q for all actions at obstacle state = -100
Q[grid == -1] = -100
r = -0.3 #Reward for each step
#ACTIONS = [(−1, 0), (1, 0), (0, −1), (0, 1)]
explored = np.zeros([40,40])
for i in range(num_episodes):
    current_state = start
    next_state = start 
    reward = 0
    
    while tuple(current_state) not in goal:
        reward = r
        #Epsilon Greedy policy
        current_state = next_state
        if random.uniform(0,1) < ep:
            action = random.choice([0,1,2,3]) #Randomly select an action up, down, left, right
        else:
            action = np.argmax(Q[current_state[0], current_state[1], :])
        
        #Move to next state
        act_dict = {0: [0,1], 1: [0,-1], 2: [1,0], 3: [-1,0]}
        next_state = current_state + act_dict[action]
        
        #Check if next state is out of bounds
        if next_state[0] < 0 or next_state[0] >= 40 or next_state[1] < 0 or next_state[1] >= 40:
            next_state = current_state
        #Check if next state is in an obstacle
        if grid[next_state[0], next_state[1]] == -1:
            next_state = current_state
            reward = -100
            
        if grid[next_state[0], next_state[1]] == 1:
            reward = 100
        
        #Bellman equation for updating Q. Q_this iteration = Q_previous iteration + alpha*(reward + gamma*max(Q_next_state) - Q_previous iteration)
        Q[current_state[0], current_state[1], action] = Q[current_state[0], current_state[1], action] + alpha*(reward+ gamma*np.max(Q[next_state[0], next_state[1], :]) - Q[current_state[0], current_state[1], action])
        explored[current_state[0], current_state[1]] = 1
print("Training Complete!")
print("Number of states explored: ", np.sum(explored))
print("Q values: ", Q)
    
#Find the path
path = []
current_state = start

while tuple(current_state) not in goal:
    action = np.argmax(Q[current_state[0], current_state[1], :])
    act_dict = {0: [0,1], 1: [0,-1], 2: [1,0], 3: [-1,0]}
    next_state = current_state + act_dict[action]
    path.append(next_state)
    current_state = next_state
    
#Plot the path
fig, ax = plt.subplots()
img = ax.imshow(grid, cmap='viridis')
for i in path:
    plt.plot(i[1], i[0], 'ro')
plt.show()

