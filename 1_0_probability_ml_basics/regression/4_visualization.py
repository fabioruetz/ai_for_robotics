import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import DataSaver as saver
from mpl_toolkits.mplot3d import Axes3D


def plot_all_variations(data_x, data_y):
    # Plots all data in 3d scatter plot
    for i in range(data_x.shape[1]):
        for k in range(i+1,data_x.shape[1]):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data_x[:,i], data_x[:,k], data_y[:])
            print('X: Data %.2f')
            print('Y: Data %.2f')


            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

            plt.show()
            plt.pause(0.5)


data_saver = saver.DataSaver('data', 'data_samples.pkl')
input_data, output_data = data_saver.restore_from_file()


plot_all_variations(input_data,output_data)