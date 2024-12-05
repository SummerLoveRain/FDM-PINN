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
from elliptic.PINN_fictitious_domain9 import PINN


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
    print('using device ' + device)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    # device = torch.device('cpu')

    # 设置随机数种子
    setup_seed(0)

    # 加载各个区域的坐标数据
    TIME_STR = '20220510_021246'
    TASK_NAME = 'task_pinn_fictitious_domain9'

    root_path = '/home/dell/yangqh/FictitiousDomain/2D_Elliptic/data/'
    # root_path = 'c:/softwarefiles/VSCodeProjects/FictitiousDomain/2D_Elliptic/data/'

    # 设置定义域
    lb = np.array([0, 0])
    ub = np.array([4, 4])
    # 椭圆区域参数
    G1 = 2
    G2 = 2
    # a = 1/4
    # b = 1/8
    a = 1
    b = 1
    def w(x1, x2): return ((x1-G1)/a)**2+((x2-G2)/b)**2 - 1

    def psi(x1, x2):
        return torch.sin(2*torch.pi*x1)*torch.cos(2*torch.pi*x2)

    # 方程参数
    alpha = 100
    mu = 0.1
    ls = 0.1
    # f(x1, x2)
    def f(x1, x2): return (alpha+8*torch.pi**2*mu)*psi(x1, x2)
    def g0(x1, x2): return psi(x1, x2)

    def g1(x1, x2, n1, n2): return mu*((n1*2*torch.pi*torch.cos(2*torch.pi*x1)*torch.cos(2*torch.pi*x2)
                                       - n2*2*torch.pi*torch.sin(2*torch.pi*x1)*torch.sin(2*torch.pi*x2))+psi(x1, x2)/ls)

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

    X = X_Pred[:, 0:1]
    Y = X_Pred[:, 1:2]
    U_true = np.sin(2*np.pi*X)*np.cos(2*np.pi*Y)
    U_flatten = U_true.flatten()[:, None]
    # 过滤，让w区域等于0
    i = 0
    for (x1, x2) in X_Pred:
        if w(x1, x2) < 0:
            U_flatten[i] = 0
        i = i + 1
    U_true = griddata(X_Pred, U_flatten.flatten(), (X, Y), method='cubic')

    scipy.io.savemat(root_path + '/' +
                     TASK_NAME + '/' + TIME_STR + '/true.mat', {'u': U_true})

    # 加载模型
    net_path = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/PINN.pkl'

    model = PINN.reload_net(net_path=net_path)
    # model.to(device)
    x1 = model.data_loader(X_Pred[:, 0:1])
    x2 = model.data_loader(X_Pred[:, 1:2])
    u_pred = model.forward(x1, x2)
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
                     TASK_NAME + '/' + TIME_STR + '/pred.mat', {'u': U_pred})

    # 存储V
    x1 = model.data_loader(X_Pred[:, 0:1])
    x2 = model.data_loader(X_Pred[:, 1:2])
    psi2 = model.net_psi2(x1, x2)
    psi2_x1 = model.compute_grad(psi2, x1)
    psi2_x2 = model.compute_grad(psi2, x2)
    psi2_x1_x1 = model.compute_grad(psi2_x1, x1)
    psi2_x2_x2 = model.compute_grad(psi2_x2, x2)
    v_pred = model.alpha * psi2 - model.mu * \
        (psi2_x1_x1 + psi2_x2_x2)
    v_pred = model.detach(v_pred)
    V_pred = griddata(X_Pred, v_pred.flatten(), (X, Y), method='cubic')

    V_flatten = V_pred.flatten()[:, None]

    # 过滤，让不在w区域等于0
    i = 0
    for (x1, x2) in X_Pred:
        if w(x1, x2) > 0:
            V_flatten[i] = 0
        i = i + 1
    V_pred = griddata(X_Pred, V_flatten.flatten(), (X, Y), method='cubic')

    scipy.io.savemat(root_path + '/' +
                     TASK_NAME + '/' + TIME_STR + '/v.mat', {'v': V_pred})

    # 存储F
    x1 = model.data_loader(X_Pred[:, 0:1])
    x2 = model.data_loader(X_Pred[:, 1:2])
    f_pred = f(x1, x2)
    f_pred = model.detach(f_pred)
    F_pred = griddata(X_Pred, f_pred.flatten(), (X, Y), method='cubic')

    F_flatten = F_pred.flatten()[:, None]

    # # 过滤，让不在w区域等于0
    # i = 0
    # for (x1, x2) in X_Pred:
    #     if w(x1, x2) > 0:
    #         V_flatten[i] = 0
    #     i = i + 1
    F_pred = griddata(X_Pred, F_flatten.flatten(), (X, Y), method='cubic')

    scipy.io.savemat(root_path + '/' +
                     TASK_NAME + '/' + TIME_STR + '/f.mat', {'f': F_pred})

    elapsed = time.time() - start_time
    print('Predicting time: %.4f' % (elapsed))
