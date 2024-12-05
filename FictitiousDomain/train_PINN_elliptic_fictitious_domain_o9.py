import datetime
import logging
import math
import os
import random
import sys
import time
import numpy as np
import torch
from elliptic.PINN_fictitious_domain9 import PINN

TASK_NAME = 'task_pinn_fictitious_domain_o9'
now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
root_path = '/home/dell/yangqh/FictitiousDomain/data/'
# root_path = 'c:/softwarefiles/VSCodeProjects/FictitiousDomain/data/'
path = '/' + TASK_NAME + '/' + now_str + '/'
log_path = root_path + '/' + path
if not os.path.exists(log_path):
    os.makedirs(log_path)
logging.basicConfig(filename=os.path.join(log_path, 'log.txt'),
                    level=logging.INFO)

# 固定随机种子，让每次运行结果一致


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    print(__file__)
    logging.info(str(__file__))

    if len(sys.argv) > 1 and 'cuda' == sys.argv[1] and torch.cuda.is_available(
    ):
        device = 'cuda'
    else:
        device = 'cpu'
    print('using device ' + device)
    logging.info('using device ' + device)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    # device = torch.device('cpu')

    # 设置随机数种子
    setup_seed(0)

    # 设置定义域
    lb = np.array([0, 0])
    ub = np.array([4, 4])

    psi1_layers = [2, 20, 20, 20, 20, 1]
    psi2_layers = [2, 20, 20, 20, 20, 1]
    v_layers = [2, 20, 20, 20, 20, 1]
    param_dict = {
        'lb': lb,
        'ub': ub,
        'psi1_layers': psi1_layers,
        'psi2_layers': psi2_layers,
        'v_layers': v_layers,
        'device': device,
        'path': path,
        'root_path': root_path,
    }

    # 椭圆区域参数
    G1 = 2
    G2 = 2
    a = 1/4
    b = 1/8
    # a = 1
    # b = 1
    def w(x1, x2): return ((x1-G1)/a)**2+((x2-G2)/b)**2 - 1

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

    # 区域网格点
    X_Omega = []
    # 内部区域网格点
    X_omega = []
    # 外边界网格点
    X_Gamma = []
    # 内边界网格点
    X_gamma = []
    for i in X:
        for j in Y:
            # 判断外边界
            if i == lb[0] or i == ub[0] or j == lb[1] or j == ub[1]:
                X_Gamma.append([i, j])
            # 判断w
            elif w(i, j) > 0:
                X_Omega.append([i, j])
            elif w(i, j) < 0:
                X_omega.append([i, j])
    # 单独再生成内边界网格点
    for i in X:
        x1 = i
        delta = (1 - ((x1-G1)/a)**2)*b**2
        if delta >= 0:
            sqrt_delta = math.sqrt(delta)
            x2_1 = sqrt_delta + G2
            x2_2 = -sqrt_delta + G2
            if x2_1 == x2_2:
                X_gamma.append([x1, x2_1])
            else:
                X_gamma.append([x1, x2_1])
                X_gamma.append([x1, x2_2])
    X_Omega = np.asarray(X_Omega, dtype=float)
    X_omega = np.asarray(X_omega, dtype=float)
    X_Gamma = np.asarray(X_Gamma, dtype=float)
    X_gamma = np.asarray(X_gamma, dtype=float)

    train_dict = {
        # 椭圆区域参数
        'G1': G1,
        'G2': G2,
        'a': a,
        'b': b,
        'w': w,
        # 方程参数
        'alpha': alpha,
        'mu': mu,
        'ls': ls,
        'f': f,
        'g0': g0,
        'g1': g1,
        'X_Omega': X_Omega,
        'X_omega': X_omega,
        'X_Gamma': X_Gamma,
        'X_gamma': X_gamma
    }
    model = PINN(param_dict=param_dict, train_dict=train_dict)
    # model.to(device)

    start_time = time.time()
    # 初始LR
    Adam_init_lr = 1e-3
    Adam_steps = 50000
    # 先用Adam 优化，再用LBFGS优化
    Adam_optimizer = torch.optim.Adam(params=model.params,
                                      lr=Adam_init_lr,
                                      betas=(0.9, 0.999),
                                      eps=1e-8,
                                      weight_decay=0,
                                      amsgrad=False)
    # Adam_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     Adam_optimizer, milestones=[10000, 100000], gamma=0.1)
    Adam_scheduler = None
    model.train_Adam(Adam_optimizer, Adam_steps, Adam_scheduler)

    # # 初始LR
    # LBFGS_init_lr = 1
    # tolerance_LBFGS = -1
    # LBFGS_steps = 10000
    # LBFGS_optimizer = torch.optim.LBFGS(
    #     params=model.params,
    #     lr=LBFGS_init_lr,
    #     max_iter=LBFGS_steps,  # max_eval=4000,
    #     tolerance_grad=tolerance_LBFGS,
    #     tolerance_change=tolerance_LBFGS,
    #     history_size=100,
    #     line_search_fn=None)

    # model.train_AdamLBFGS(Adam_optimizer, None, LBFGS_optimizer, Adam_steps)

    # 打印总耗时
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    logging.info('Training time: %.4f' % (elapsed))
