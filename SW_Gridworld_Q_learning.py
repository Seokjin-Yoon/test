# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:14:21 2024

@author: Seokjin
"""

import numpy as np
import matplotlib.pyplot as plt
import random

class gridWorld(object):
    
    def __init__(self):
        super(gridWorld, self).__init__()
        self.start = 0
        self.goal = 0
        self.row = 7
        self.col = 10
        self.x_max = self.col - 1
        self.y_max = self.row - 1
        self.wind_1 = [3, 4, 5, 8]
        self.wind_2 = [6, 7]
        self.actions_list = ['N','E','S','W','NE','NW','SE','SW','X']
        
    def cell(self,pos):
        return pos[1] + self.col * pos[0]
    
    def setTerminal(self, startState, goalState):
        self.start = self.cell(startState)
        self.goal = self.cell(goalState)
        
    def nextState(self, state, action):
        x = state % self.col
        y = (state - x) / self.col
        del_x = 0
        del_y = 0
        if action == 'E':
            del_x = 1
        elif action == 'W':
            del_x = -1
        elif action == 'N':
            del_y = -1
        elif action == 'S':
            del_y = 1
        elif action == 'NE':
            del_x ,del_y = 1, 1
        elif action == 'NW':
            del_x, del_y = 1, -1
        elif action == 'SE':
            del_x, del_y = -1, 1
        elif action == 'SW':
            del_x, del_y = -1, -1
        elif action == 'X':
            del_x, del_y = 0, 0
        else:
            raise('Invalid action! Actions taken must be in: ',self.actions_list)
        new_x = max(0, min(x + del_x, self.x_max))
        new_y = max(0, min(y + del_y, self.y_max))
        if new_x in self.wind_1:
            prob = random.randrange(3)
            new_y += prob - 1
            new_y = max(0, min(new_y, self.y_max))
        if new_x in self.wind_2:
            prob = random.randrange(3)
            new_y += 2*(prob - 1)
            new_y = max(0, min(new_y, self.y_max))
        return self.cell((new_y,new_x))
    
    def checkTerminal(self, state):
        return state == self.goal
    
    def rewardFunction(self, state_prime):
        if state_prime == self.goal:
            return 0
        else:
            return -1
        
def trajectoryPath(world, traj):
    world_map = np.zeros((world.row, world.col))
    for i,state in enumerate(traj):
        x = int(state % world.col)
        y = int((state - x) / world.col)
        world_map[y, x] = i + 1
    print(world_map)
    print("\n")

def gridWorld_QLearning(world, startState, goalState, alpha, gamma=1, ep_max=10000, eps=0.1):
    world.setTerminal(startState, goalState) 
    q_dict = {}
    for state in range(world.row * world.col):
        q_dict[state] = {}
        for act in world.actions_list:
                q_dict[state][act] = 0

    def greedyAct(_q_dict):
        greedy_act = ''
        max_q = -1e10
        for act in world.actions_list:
            if _q_dict[act] > max_q:
                greedy_act = act
                max_q = _q_dict[act]
        return greedy_act

    def epsGreedy(episode, q_dict):
        m = len(world.actions_list)
        greedy_act = greedyAct(q_dict)
        p = []
        for act in world.actions_list:
            if act == greedy_act:
                p.append((eps * 1. / m) + 1 - eps)
            else:
                p.append(eps * 1. / m)
        choice = np.random.choice(world.actions_list, size=1, p=p)
        return choice[0]

    ep_wrt_step = []
    trajectory = []
    for ep in range(1, ep_max + 1):
        s = world.start
        trajectory = []
        while not world.checkTerminal(s):
            act = epsGreedy(ep, q_dict[s])
            s_prime = world.nextState(s, act)
            reward = world.rewardFunction(s_prime)
            
            act_prime = greedyAct(q_dict[s_prime])
            q_dict[s][act] += alpha * (reward + gamma * q_dict[s_prime][act_prime] - q_dict[s][act])
            
            trajectory.append(s)
            s = s_prime
            ep_wrt_step.append(ep)
        trajectory.append(world.goal)
    return trajectory, ep_wrt_step


startState = (3, 0)
goalState = (3, 7)
world = gridWorld()
trajectory, ep_wrt_step = gridWorld_QLearning(world, startState, goalState, alpha=0.5,gamma=0.9,ep_max=10000, eps=0.1)
trajectoryPath(world, trajectory)
plt.figure(1)
plt.plot(ep_wrt_step)
plt.xlabel("Time steps")
plt.ylabel("Episodes")
plt.show()