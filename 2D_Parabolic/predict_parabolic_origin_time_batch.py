import datetime
import logging
import math
import os
import random
import sys
import time
import numpy as np
import scipy.io
from scipy.interpolate import griddata
import torch
from parabolic.PINN_origin_time_batch import PINN


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    start_time = time.time()
    if len(sys.argv) > 1 and 'cuda' == sys.argv[1] and torch.cuda.is_available(
    ):
        device = 'cuda'
    else:
        device = 'cpu'
    if len(sys.argv) > 2:
        PRED_T = float(sys.argv[2])
    else:
        PRED_T = float(1)
    print('using device ' + device)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    # device = torch.device('cpu')

    # 设置随机数种子
    setup_seed(0)

    # 加载各个区域的坐标数据
    TIME_STR = '20220512_170904'
    TASK_NAME = 'task_pinn_origin_time_batch'

    root_path = '/home/dell/yangqh/FictitiousDomain/2D_Parabolic/data/'
    # root_path = 'c:/softwarefiles/VSCodeProjects/FictitiousDomain/2D_Parabolic/data/'

    PRED_T = 0.5
    # 设置定义域
    lb = np.array([0, 0, 1])
    ub = np.array([4, 4, 1])
    # 椭圆区域参数
    G1 = 2
    G2 = 2
    a = 1/4
    b = 1/8
    def w(x1, x2): return ((x1-G1)/a)**2+((x2-G2)/b)**2 - 1

    # 方程参数
    alpha = 100
    mu = 0.1
    ls = 0.1

    def psi(x1, x2, t): return (x1**3-x2**3)*t

    def psi0(x1, x2): return 0

    def f(x1, x2, t): return (x1**3-x2**3)-6*mu*(x1-x2)*t
    def g0(x1, x2, t): return psi(x1, x2, t)

    def g1(x1, x2, t, n1, n2): return mu * \
        (3*(n1*x1**2-n2*x2**2)+(x1**3-x2**3)/ls)*t

    # 生成网格点
    GRID_SIZE = 100

    X = [i for i in range(GRID_SIZE * (ub[0] - lb[0]) + 1)]
    X = np.asarray(X, dtype=float)
    X = X / GRID_SIZE
    Y = [i for i in range(GRID_SIZE * (ub[1] - lb[1]) + 1)]
    Y = np.asarray(Y, dtype=float)
    Y = Y / GRID_SIZE

    # 预测网格点
    X_Pred = []
    X, Y = np.meshgrid(X, Y)
    X_Pred = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))  # 组合了X和T

    T = np.ones_like(X)*PRED_T

    U_true = psi(X, Y, T)
    U_flatten = U_true.flatten()[:, None]

    # 过滤，让w区域等于0
    i = 0
    for (x1, x2) in X_Pred:
        if w(x1, x2) < 0:
            U_flatten[i] = 0
        i = i + 1
    U_true = griddata(X_Pred, U_flatten.flatten(), (X, Y), method='cubic')
    
    scipy.io.savemat(root_path + '/' +
                     TASK_NAME + '/' + TIME_STR + '/true_T='+str(PRED_T)+'.mat', {'u': U_true})
    # 加载模型
    net_path = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/PINN.pkl'

    model = PINN.reload_net(net_path=net_path)
    # model.to(device)
    x1 = model.data_loader(X_Pred[:, 0:1])
    x2 = model.data_loader(X_Pred[:, 1:2])
    t = np.ones_like(X_Pred[:, 0:1])*PRED_T
    t = model.data_loader(t)
    with torch.no_grad():
        u_pred = model.forward(x1, x2, t)
    u_pred = model.detach(u_pred)
    U_pred = griddata(X_Pred, u_pred.flatten(), (X, Y), method='cubic')

    U_flatten = U_pred.flatten()[:, None]

    # 过滤，让w区域等于0
    i = 0
    for (x1, x2) in X_Pred:
        if w(x1, x2) < 0:
            U_flatten[i] = 0
        i = i + 1
    U_pred = griddata(X_Pred, U_flatten.flatten(), (X, Y), method='cubic')

    scipy.io.savemat(root_path + '/' +
                     TASK_NAME + '/' + TIME_STR + '/pred_T='+str(PRED_T)+'.mat', {'u': U_pred})
    # plot_contour(X, Y, U_pred, title=None, file_name=root_path + '/' +
    #                 TASK_NAME + '/pred_' + TIME_STR)

    U_error = U_true - U_pred
    scipy.io.savemat(root_path + '/' +
                     TASK_NAME + '/' + TIME_STR + '/error.mat', {'u': U_error})

    elapsed = time.time() - start_time
    print('Predicting time: %.4f' % (elapsed))
