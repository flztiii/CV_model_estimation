# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:43:06 2019

@author: flztiii
"""

import EKF
import UKF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

METHOD_TYPE = "UKF"
REPEAT_TIMES = 1
COUNTS = 50

if __name__ == "__main__":
    predictions = []
    ground_truthes = []
    estimations = []
    # get simulated data
    if METHOD_TYPE == "EKF":
        init_pose = np.array([1,1,1,1,1,1])
        init_P = np.eye(6)
        for iteration in range(0, REPEAT_TIMES):
            ekfilter = EKF.EKFilter(init_pose, init_P)
            ekfilter.update(COUNTS)
            ground_truthes.append(ekfilter.getTrueData())
            predictions.append(ekfilter.getPrediction())
            estimations.append(ekfilter.getEstimation())
    elif METHOD_TYPE == "UKF":
        init_pose = np.array([[1,1,1,1,1,1]])
        init_P = np.eye(6)
        for iteration in range(0, REPEAT_TIMES):
            ukfilter = UKF.UKFilter(init_pose, init_P)
            ukfilter.update(COUNTS)
            ground_truthes.append(ukfilter.getTrueData())
            predictions.append(ukfilter.getPrediction())
            estimations.append(ukfilter.getEstimation())
    
    # plot data
    for i in range(0, len(ground_truthes)):
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        ground_truth_pos = []
        prediction_pos = []
        estimation_pos = []
        for j in range(0, len(ground_truthes[i])):
            ground_truth_pos.append([ground_truthes[i][j][0], ground_truthes[i][j][2], ground_truthes[i][j][4]])
            prediction_pos.append([predictions[i][j][0], predictions[i][j][2], predictions[i][j][4]])
            estimation_pos.append([estimations[i][j][0], estimations[i][j][2], estimations[i][j][4]])
        ground_truth_pos = np.array(ground_truth_pos)
        prediction_pos = np.array(prediction_pos)
        estimation_pos = np.array(estimation_pos)
        ax.plot3D(ground_truth_pos[:,0],ground_truth_pos[:,1],ground_truth_pos[:,2],'red')
        ax.plot3D(estimation_pos[:,0],estimation_pos[:,1],estimation_pos[:,2],'green')
        ax.plot3D(prediction_pos[:,0],prediction_pos[:,1],prediction_pos[:,2],'blue')
        plt.show()