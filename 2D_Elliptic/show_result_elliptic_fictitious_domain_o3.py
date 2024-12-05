import math
import os
import time
from matplotlib import cm, pyplot as plt
import numpy as np
import scipy.io
from plot.heatmap import plot_heatmap, plot_heatmap3
from plot.surface import plot_surface_3D

if __name__ == "__main__":

    start_time = time.time()
    # 加载各个区域的坐标数据
    TIME_STR = '20220510_150741'
    TASK_NAME = 'task_pinn_fictitious_domain_o3'

    root_path = '/home/dell/yangqh/FictitiousDomain/2D_Elliptic/data/'
    # root_path = 'c:/softwarefiles/VSCodeProjects/FictitiousDomain/2D_Elliptic/data/'

    # 设置定义域
    lb = np.array([0, 0])
    ub = np.array([4, 4])
    # 椭圆区域参数
    G1 = 2
    G2 = 2
    a = 1/4
    b = 1/8
    # a = 1
    # b = 1
    def w(x1, x2): return ((x1-G1)/a)**2+((x2-G2)/b)**2 - 1

    def psi(x1, x2):
        return x1**3-x2**3

    # 方程参数
    alpha = 100
    mu = 0.1
    ls = 0.1
    # f(x1, x2)
    def f(x1, x2): return alpha*(x1**3-x2**3)-6*mu*(x1-x2)
    def g0(x1, x2): return x1**3-x2**3
    def g1(x1, x2, n1, n2): return mu*(3*(n1*x1**2-n2*x2**2)+(x1**3-x2**3)/ls)

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
    u_true = data['u']
    u_true = u_true.reshape((X.size, Y.size))

    data = scipy.io.loadmat(root_path + '/' +
                            TASK_NAME + '/' + TIME_STR + '/pred.mat')
    u_pred = data['u']
    u_pred = u_pred.reshape((X.size, Y.size))
    file_name = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/heatmap3'

    plot_heatmap3(X, Y, u_true, u_pred, E=None, xlabel='x1',
                  ylabel='x2', file_name=file_name)

    # 画V
    data = scipy.io.loadmat(root_path + '/' +
                            TASK_NAME + '/' + TIME_STR + '/v.mat')
    v_true = data['v']
    v_true = v_true.reshape((X.size, Y.size))
    plot_surface_3D(X, Y, v_true, title='v', xlabel='x1', ylabel='x2', zlabel='v', file_name=root_path + '/' +
                    TASK_NAME + '/' + TIME_STR + '/v')
    # 画F
    data = scipy.io.loadmat(root_path + '/' +
                            TASK_NAME + '/' + TIME_STR + '/f.mat')
    f_true = data['f']
    f_true = f_true.reshape((X.size, Y.size))
    plot_surface_3D(X, Y, f_true, title='f', xlabel='x1', ylabel='x2', zlabel='f', file_name=root_path + '/' +
                    TASK_NAME + '/' + TIME_STR + '/f')

    # plt.show()
    elapsed = time.time() - start_time
    print('Predicting time: %.4f' % (elapsed))
