# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:42:22 2019

@author: flztiii
"""

# data format: [[x, vx, y, vy, z, vz],[data_2],...,[data_n]]

import numpy as np
import sympy
import random


class EKFilter:
    def __init__(self, init_pose, init_P):
        self.true_data = []
        self.observe_data = []
        self.predict_data = []
        self.estimate_data = []
        self.sigma_w = 0.5
        self.sigma_v = 0.2
        self.init_pose = np.array(init_pose)
        self.P = np.array(init_P)
        self.T = 1.0
        self.Q = np.diag([self.sigma_w, self.sigma_w, self.sigma_w])
        self.R = np.diag([self.sigma_v, self.sigma_v, self.sigma_v])
        self.CVModelBuild()
        self.initPose()
        
    def CVModelBuild(self):
        self.f_x = np.array([[1,self.T,0,0,0,0],[0,1,0,0,0,0],[0,0,1,self.T,0,0],[0,0,0,1,0,0], [0,0,0,0,1,self.T], [0,0,0,0,0,1]])
        self.f_w = np.array([[self.T*self.T/2.0,0,0],[self.T,0,0],[0,self.T*self.T/2.0,0],[0,self.T,0],[0,0,self.T*self.T/2.0],[0,0,self.T]])
        x, vx, y, vy, z, vz= sympy.symbols('x vx y vy z vz')
        self.vari = [x, vx, y, vy, z, vz]
        o1 = sympy.sqrt(x*x+y*y+z*z)
        o2 = sympy.atan(y/x)
        o3 = sympy.atan(z/sympy.sqrt(x*x+y*y))
        self.H = [o1, o2, o3]
        self.h_x = []
        for i in range(0, len(self.H)):
            h_x_row = []
            for j in range(0, len(self.vari)):
                h_x_row.append(sympy.diff(self.H[i], self.vari[j]))
            self.h_x.append(h_x_row)
        self.h_v = np.array([[1,0,0],[0,1,0],[0,0,1]])
    
    def initPose(self):
        self.true_x = np.array(self.init_pose)
        self.estimate_x = np.array(self.init_pose)
        self.true_data.append(self.init_pose)
        self.predict_data.append(self.init_pose)
        self.estimate_data.append(self.init_pose)
        self.observe_data.append([float(self.H[0].subs([(self.vari[0], self.init_pose[0]),(self.vari[2], self.init_pose[2]),(self.vari[4], self.init_pose[4])])), float(self.H[1].subs([(self.vari[0], self.init_pose[0]),(self.vari[2], self.init_pose[2]),(self.vari[4], self.init_pose[4])])), float(self.H[2].subs([(self.vari[0], self.init_pose[0]),(self.vari[2], self.init_pose[2]),(self.vari[4], self.init_pose[4])]))])
    
    def update(self, count):
        for k in range(0, count):
            w = np.array([random.gauss(0, self.sigma_w), random.gauss(0, self.sigma_w), random.gauss(0, self.sigma_w)])
            v = np.array([random.gauss(0, self.sigma_v), random.gauss(0, self.sigma_v), random.gauss(0, self.sigma_v)])
            self.true_x = np.dot(self.f_x, self.true_x.T)+np.dot(self.f_w,w.T)
            # print(self.true_x)
            observe_x = np.array([float(self.H[0].subs([(self.vari[0], self.true_x[0]),(self.vari[2], self.true_x[2]),(self.vari[4], self.true_x[4])])), float(self.H[1].subs([(self.vari[0], self.true_x[0]),(self.vari[2], self.true_x[2]),(self.vari[4], self.true_x[4])])), float(self.H[2].subs([(self.vari[0], self.true_x[0]),(self.vari[2], self.true_x[2]),(self.vari[4], self.true_x[4])]))])+v
            # print(observe_x)
            predict_x = np.dot(self.f_x, self.estimate_x.T)
            # print(predict_x)
            P_pre = np.dot(np.dot(self.f_x, self.P), self.f_x.T)+np.dot(np.dot(self.f_w, self.Q), self.f_w.T)
            # print(P_pre)
            h_x_value = np.zeros((3, 6))
            for i in range(0, 3):
                for j in range(0, 6):
                    h_x_value[i, j]= float(self.h_x[i][j].subs([(self.vari[0], predict_x[0]),(self.vari[2], predict_x[2]),(self.vari[4], predict_x[4])]))
            # print(h_x_value)
            K = np.dot(np.dot(P_pre, h_x_value.T), np.linalg.inv(np.dot(np.dot(h_x_value, P_pre), h_x_value.T)+np.dot(np.dot(self.h_v, self.R), self.h_v.T)))
            
            self.estimate_x = predict_x + np.dot(K, observe_x - np.dot(h_x_value, predict_x))
            # print(self.estimate_x)
            self.P = P_pre - np.dot(np.dot(K, h_x_value), P_pre)
            
            self.true_data.append(self.true_x)
            self.predict_data.append(predict_x)
            self.observe_data.append(observe_x)
            self.estimate_data.append(self.estimate_x)
        
        def getTrueData(self):
            return self.true_data
        
        def getObservation(self):
            return self.observe_data
        
        def getPrediction(self):
            return self.predict_data
        
        def getEstimation(self):
            return self.estimate_data

if __name__ == "__main__":
    ekfilter = EKFilter([10,1,10,1,10,1],1)
    ekfilter.update(1)