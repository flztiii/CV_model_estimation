# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:42:48 2019

@author: flztiii
"""

import numpy as np
import random

class UKFilter:
    def __init__(self, init_pose, init_P):
        self.true_data = []
        self.observe_data = []
        self.predict_data = []
        self.estimate_data = []
        self.sigma_w = 0.5*0.5
        self.sigma_v = 0.01*0.01
        self.true_x = np.array(init_pose)
        self.estimate_x = np.array(init_pose)
        self.P = np.array(init_P)
        self.dt = 0.1
        self.f_w = np.array([[self.dt*self.dt/2.0,0,0],[self.dt,0,0],[0,self.dt*self.dt/2.0,0],[0,self.dt,0],[0,0,self.dt*self.dt/2.0],[0,0,self.dt]])
        self.Q = np.dot(np.dot(self.f_w, np.diag([self.sigma_w, self.sigma_w, self.sigma_w])), self.f_w.T)
        self.R = np.diag([self.sigma_v, self.sigma_v, self.sigma_v])
        
    def calcSigmaPoints(self, x, sigma, alpha=.3, beta=2., kappa=.1):
        points = []
        weights_m = []
        weights_c = []
        n = x.size
        lamb = float(alpha**2*(n + kappa) - n)
        for i in range(0, 2*n+1):
            if i == 0:
                point = x
                w_m = lamb/float(n + lamb)
                w_c = lamb/float(n + lamb) + 1 - alpha**2 + beta
            elif i < n+1:
                point = x + np.linalg.cholesky((n + lamb)*sigma)[:,i-1].T
                w_m = 1.0/(2.0*(n + lamb))
                w_c = 1.0/(2.0*(n + lamb))
            else:
                point = x - np.linalg.cholesky((n + lamb)*sigma)[:,i-n-1].T
                w_m = 1.0/(2.0*(n + lamb))
                w_c = 1.0/(2.0*(n + lamb))
            points.append(point)
            weights_m.append(w_m)
            weights_c.append(w_c)
        return np.array(points), np.array(weights_m), np.array(weights_c)
    
    def F(self, x):
        f_x = np.array([[1,self.dt,0,0,0,0],[0,1,0,0,0,0],[0,0,1,self.dt,0,0],[0,0,0,1,0,0], [0,0,0,0,1,self.dt], [0,0,0,0,0,1]])
        return np.dot(f_x, x.T).T
    
    def H(self, x):
        px = x[0]
        py = x[2]
        pz = x[4]
        z = np.array([[np.sqrt(px**2 + py**2 + pz**2), np.arctan(py/px), np.arctan(pz/np.sqrt(px**2+py**2))]])
        return z
    
    def unscentedTransform(self, points, weights_m, weights_c, matrix):
        x = np.zeros_like(points[0])
        sigma = matrix
        for i in range(0, len(points)):
            x = x + weights_m[i]*points[i]
        for i in range(0, len(points)):
            sigma = sigma + weights_c[i]*np.dot((points[i] - x).T,(points[i] - x))
        return x, sigma
    
    def predict(self):
        x_points, self.ws_m, self.ws_c = self.calcSigmaPoints(self.estimate_x, self.P)
        print(self.ws_m)
        print(self.ws_c)
        print(x_points)
        self.trans_x_points = np.zeros_like(x_points)
        for i in range(0, len(x_points)):
            self.trans_x_points[i] = self.F(x_points[i])
        self.x_pre, self.P_pre = self.unscentedTransform(self.trans_x_points, self.ws_m, self.ws_c, self.Q)
        print(self.trans_x_points)
        print(self.x_pre)
        
    def correct(self):
        self.trans_z_points = []
        for i in range(0, len(self.trans_x_points)):
            self.trans_z_points.append(self.H(self.trans_x_points[i][0]))
        self.trans_z_points = np.array(self.trans_z_points)
        mu_z, P_z = self.unscentedTransform(self.trans_z_points, self.ws_m, self.ws_c, self.R)
        y = self.observe_x - mu_z
        P_xz = np.zeros((6, 3))
        for i in range(0, len(self.trans_z_points)):
            P_xz = P_xz + self.ws_c[i]*np.dot((self.trans_x_points[i] - self.x_pre).T, self.trans_z_points[i] - mu_z)
        K = np.dot(P_xz, np.linalg.inv(P_z))
        self.estimate_x = self.x_pre + np.dot(K, y.T).T
        self.P = self.P_pre - np.dot(np.dot(K, P_z), K.T)
        
    def update(self, count):
        for k in range(0, count):
            w = np.array([[random.gauss(0, self.sigma_w), random.gauss(0, self.sigma_w), random.gauss(0, self.sigma_w)]])
            v = np.array([[random.gauss(0, self.sigma_v), random.gauss(0, self.sigma_v), random.gauss(0, self.sigma_v)]])
            self.true_x = self.F(self.true_x) + np.dot(self.f_w, w.T).T
            self.observe_x = self.H(self.true_x[0]) + v
            self.predict()
            self.correct()
            self.true_data.append(self.true_x[0])
            self.observe_data.append(self.observe_x[0])
            self.predict_data.append(self.x_pre[0])
            self.estimate_data.append(self.estimate_x[0])
    
    def getTrueData(self):
        return self.true_data
    
    def getObservation(self):
        return self.observe_data
    
    def getPrediction(self):
        return self.predict_data
    
    def getEstimation(self):
        return self.estimate_data

if __name__ == "__main__":
    ukfilter = UKFilter(np.array([[1,1,1,1,1,1]]),np.eye(6));
    ukfilter.update(1)