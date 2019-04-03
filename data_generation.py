# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:38:59 2019

@author: flztiii
"""

# data format: [[x, y, vx, vy],[data_2],...,[data_n]]

import numpy as np
import random

class DataGenerator:
    def __init__(self, routine_type="line"):
        self.routine_type = routine_type
        self.sigma_w = 0.3
        self.sigma_v = 0.3
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        self.gamma = np.array([[0.5,0],[0,0.5],[1,0],[0,1]])
        self.trueValueObtain()
        self.addNoise()
        
    def trueValueObtain(self):
        self.true_data = []
        self.true_observation = []
        if self.routine_type == "line":
            for x in range(10,100,1):
                y = x
                vx = 1
                vy = 1
                self.true_data.append([x, y, vx, vy])
                self.true_observation.append([float(y)/float(x), float(x*x+y*y)])
        elif self.routine_type == "curve":
            for x in range(10,100,1):
                y = 100 - np.sqrt(100*100 - x*x)
                vx = 1
                vy = float((100 - np.sqrt(100*100 - (x + vx)*(x + vx))) - (100 - np.sqrt(100*100 - (x - vx)*(x - vx))))/2.0
                self.true_data.append([x, y, vx, vy])
                self.true_observation.append([float(y)/float(x), float(x*x+y*y)])
        else:
            print("[ERROE] error routine type")
        self.true_data = np.array(self.true_data)
        self.true_observation = np.array(self.true_observation)
    
    def addNoise(self):
        self.prediction = []
        self.observation = []
        for i in range(0, len(self.true_data)):
            if i == 0:
                self.prediction.append(self.true_data[i])
            else:
                w_x = random.gauss(0, self.sigma_w)
                w_y = random.gauss(0, self.sigma_w)
                w = np.array([w_x, w_y])
                pre_data = np.dot(self.F, self.true_data[i].T) + np.dot(self.gamma, w.T)
                self.prediction.append(pre_data)
        self.prediction = np.array(self.prediction)
        # print(self.prediction)
        for i in range(0, len(self.true_observation)):
            v_alpha = random.gauss(0, self.sigma_v)
            v_beta = random.gauss(0, self.sigma_v)
            self.observation.append([self.true_observation[i][0]+v_alpha, self.true_observation[i][1]+v_beta])
        self.observation = np.array(self.observation)
    
    def getTrueData(self):
        return self.true_data
        
    def getPrediction(self):
        return self.prediction
    
    def getObservation(self):
        return self.observation
    
    def getInitialData(self):
        return self.true_data[0]


if __name__ == "__main__":
    data_generator = DataGenerator() 