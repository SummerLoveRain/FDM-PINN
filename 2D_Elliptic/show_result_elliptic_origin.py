import os
import time
from matplotlib import cm, pyplot as plt
import numpy as np
import scipy.io

from plot.heatmap import plot_heatmap, plot_heatmap3

if __name__ == "__main__":

    start_time = time.time()
    # 加载各个区域的坐标数据
    TIME_STR = '20220510_021346'
    TASK_NAME = 'task_pinn_origin'

    root_path = 'c:/softwarefiles/VSCodeProjects/FictitiousDomain/2D_Elliptic/data/'

    # 设置定义域
    lb = np.array([0, 0])
    ub = np.array([4, 4])

    # 生成网格点
    GRID_SIZE = 100

    X = [i for i in range(GRID_SIZE * (ub[0] - lb[0]) + 1)]
    X = np.asarray(X, dtype=float)
    X = X / GRID_SIZE
    Y = [i for i in range(GRID_SIZE * (ub[1] - lb[1]) + 1)]
    Y = np.asarray(Y, dtype=float)
    Y = Y / GRID_SIZE

    data = scipy.io.loadmat(root_path + '/' +
                            TASK_NAME + '/' + TIME_STR + '/true.mat')
    u_true = data['u'].flatten()[:, None]
    u_true = u_true.reshape((X.size, Y.size))
    # plot_heatmap(X, Y, u_true, title='True', file_name=root_path + '/' +
    #              TASK_NAME + '/' + TIME_STR + '/true')

    data = scipy.io.loadmat(root_path + '/' +
                            TASK_NAME + '/' + TIME_STR + '/pred.mat')
    u_pred = data['u'].flatten()[:, None]
    u_pred = u_pred.reshape((X.size, Y.size))
    # plot_heatmap(X, Y, u_pred, title='Pred', file_name=root_path + '/' +
    #              TASK_NAME + '/' + TIME_STR + '/pred')

    # data = scipy.io.loadmat(root_path + '/' +
    #                         TASK_NAME + '/' + TIME_STR + '/error.mat')
    # u_pred = data['u'].flatten()[:, None]
    # u_pred = u_pred.reshape((X.size, Y.size))
    # plot_heatmap(X, Y, u_pred, title='Error', file_name=root_path + '/' +
    #              TASK_NAME + '/' + TIME_STR + '/error')
    # plt.show()

    file_name = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/heatmap3'
    plot_heatmap3(X, Y, u_true, u_pred, E=None, xlabel='x1',
                  ylabel='x2', file_name=file_name)

    elapsed = time.time() - start_time
    print('Predicting time: %.4f' % (elapsed))
