'''
Task: to navigate a 40x40 square grid from 'S' to 'G' using Q learning
Assumptions:
1. The agent can move in 4 directions - up, down, left, right
2. The agent can move only to the neighbouring grids

Hyperparameters:
1. Reward = -1 for each step to find shortest path
2. Value of goal state = 100
3. Value of obstacle = -100

5. epsilon
6. gamma
7. alpha

For creating the Q table we will use epsilon greedy policy
'''


import numpy as np
import matplotlib.pyplot as plt
import random


def dist_to_goal(a, goal):
    #min distance to all the goal states
    return min([np.linalg.norm(np.array(a)-np.array(i)) for i in goal])

with open(r"C:\Users\ganga\Documents\IISc Coursework\2nd Sem\Motion Planning\grid\MotionPlanning_2025\grid.txt") as f:
    grid = f.read().splitlines()

#Convert '.' to 0, 'X' to 1, 'S' to 0, 'G' to 100
grid_dict = {'.': 0, 'X': -1, 'S': 2, 'G': 1}
grid = np.array([[grid_dict[j] for j in i.split()] for i in grid])

'''#plot the grid
fig, ax = plt.subplots()
img = ax.imshow(grid, cmap='viridis')
plt.show()
'''
start = np.where(grid == 2)
goal = np.where(grid == 1)

start = np.array([start[0][0], start[1][0]])  # Convert to (row, col)
goal = list(zip(goal[0], goal[1]))  # List of all goal coordinates


#Stochastic, with actions as up, down, left, right, and diagonals variable greedy epsilon policy, #Initialised Q values based on distance to goal#

#Initialize a 40 x 40 x4 grid Q values: each grid has 4 possible actions
Q = np.zeros((40,40,8))
ep = 0.9 #Epsilon for epsilon greedy policy
ep_decay = 0.99 #Decay rate for epsilon
ep_min = 0.01 #Minimum epsilon
gamma = 0.9 #Discount factor
alpha = 0.1 #Learning rate
num_episodes = 8000
grid_size = grid.shape
prob_action = 0.7
r = -0.3
#Make an array to indicate if a grid has been explored or not
explored = np.zeros(grid_size)

# Set goal state Q-values
for g in goal:
    Q[g[0], g[1], :] = 100  # Assign high reward to all actions at goal states
    
# Set obstacle states Q-values
Q[grid == -1, :] = -100  # Ensure grid indexing is correct

# Map the distance to the goal as a reward function
max_dist = dist_to_goal(start, goal)

# Assign rewards based on distance to goal
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        if grid[i, j] == 0 or grid[i,j] ==2:  # Ignore obstacles
            dist = dist_to_goal((i, j), goal)
            Q[i, j, :] = (max_dist - dist) * (10 / max_dist)
            
for i in range(num_episodes):
    current_state = start
    next_state = start
    reward = 0
    while tuple(current_state) not in goal:
        reward = r
        #Epsilon Greedy policy
        current_state = next_state
        if random.uniform(0,1) < ep:
            action = random.choice([0,1,2,3,4,5,6,7]) #Randomly select an action up, down, left, right, up-right, up-left, down-right, down-left
        else:
            action = np.random.choice(np.flatnonzero(Q[current_state[0], current_state[1], :] == Q[current_state[0], current_state[1], :].max()))
        
        #Move to next state
        act_dict = {0: [0,1], 1: [0,-1], 2: [1,0], 3: [-1,0], 4: [1,1], 5: [1,-1], 6: [-1,1], 7: [-1,-1]}#0 is right, 1 is left, 2 is down, 3 is up
        if random.uniform(0,1) < prob_action:
            next_state = current_state + act_dict[action]
        else:
            next_state = current_state + random.choice(list(act_dict.values()))
        
        #Check if next state is out of bounds
        if next_state[0] < 0 or next_state[0] >= 40 or next_state[1] < 0 or next_state[1] >= 40:
            next_state = current_state
            #reward = -20
        #Check if next state is in an obstacle
        if grid[next_state[0], next_state[1]] == -1:
            next_state = current_state
            reward = -100
        elif grid[next_state[0], next_state[1]] == 1:
            reward = 100
        elif dist_to_goal(next_state, goal) < dist_to_goal(current_state, goal):
            reward = reward + 10
            
        #Bellman equation for updating Q. Q_this iteration = Q_previous iteration + alpha*(reward + gamma*max(Q_next_state) - Q_previous iteration)
        Q[current_state[0], current_state[1], action] = Q[current_state[0], current_state[1], action] + alpha*(reward+ gamma*np.max(Q[next_state[0], next_state[1], :]) - Q[current_state[0], current_state[1], action])
        
        explored[current_state[0], current_state[1]] = 1
    ep = max(ep*ep_decay, ep_min)
print("Training Complete!")
print("Q values: ", Q)

#Find the path
path = []
current_state = start
action = np.argmax(Q[current_state[0], current_state[1], :])
#print(action)


while tuple(current_state) not in goal:
    action = np.argmax(Q[current_state[0], current_state[1], :])
    #print(action)
    #act_dict = {0: [0,1], 1: [0,-1], 2: [1,0], 3: [-1,0]}
    act_dict = {0: [0,1], 1: [0,-1], 2: [1,0], 3: [-1,0], 4: [1,1], 5: [1,-1], 6: [-1,1], 7: [-1,-1]}
    next_state = current_state + act_dict[action]
    #print(next_state)
    path.append(next_state)
    current_state = next_state
    #if stuck in a loop break
    if len(path) > 2000:
        print("Stuck in a loop")
        break
    
#Plot the path

fig, ax = plt.subplots()
img = ax.imshow(grid, cmap='viridis')
# Highlight explored states with a semi-transparent overlay
explored_overlay = np.where(explored, 0.5, np.nan)  # 0.5 for semi-transparent effect
ax.imshow(explored_overlay, cmap='Blues', alpha=0.3)  # Blue for explored states
for i in path:
    plt.plot(i[1], i[0], 'ro')
plt.show()

print("Number of states explored: ", np.sum(explored))