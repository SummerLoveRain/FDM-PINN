import datetime
import logging
import math
import os
import random
import sys
import time
import numpy as np
import torch
from parabolic.PINN_fictitious_time_batch import PINN

TASK_NAME = 'task_pinn_fictitious_time_batch'
now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
root_path = '/home/dell/yangqh/FictitiousDomain/2D_Parabolic/data/'
# root_path = 'c:/softwarefiles/VSCodeProjects/FictitiousDomain/2D_Parabolic/data/'
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
    lb = np.array([0, 0, 0])
    ub = np.array([4, 4, 1])

    psi1_layers = [3, 80, 80, 80, 80, 1]
    psi2_layers = [3, 40, 40, 40, 40, 1]
    param_dict = {
        'lb': lb,
        'ub': ub,
        'psi1_layers': psi1_layers,
        'psi2_layers': psi2_layers,
        'device': device,
        'path': path,
        'root_path': root_path,
    }

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
    Omega_GRID_SIZE = 40
    omega_GRID_SIZE = 100
    delta_T = 10

    Omega_X = [i for i in range(Omega_GRID_SIZE * (ub[0] - lb[0]) + 1)]
    Omega_X = np.asarray(Omega_X, dtype=float)
    Omega_X = Omega_X / Omega_GRID_SIZE
    Omega_Y = [i for i in range(Omega_GRID_SIZE * (ub[1] - lb[1]) + 1)]
    Omega_Y = np.asarray(Omega_Y, dtype=float)
    Omega_Y = Omega_Y / Omega_GRID_SIZE

    omega_X = [i for i in range(omega_GRID_SIZE * (ub[0] - lb[0]) + 1)]
    omega_X = np.asarray(omega_X, dtype=float)
    omega_X = omega_X / omega_GRID_SIZE
    omega_Y = [i for i in range(omega_GRID_SIZE * (ub[1] - lb[1]) + 1)]
    omega_Y = np.asarray(omega_Y, dtype=float)
    omega_Y = omega_Y / omega_GRID_SIZE

    T = [i for i in range(delta_T * (ub[2] - lb[2]) + 1)]
    T = np.asarray(T, dtype=float)
    T = T / delta_T

    # 区域网格点
    X_Omega = []
    # 内部区域网格点
    X_omega = []
    # 内边界网格点
    X_gamma = []
    for t in T:
        if t == 0:
            continue
        for i in Omega_X:
            for j in Omega_Y:
                if i != lb[0] and i != ub[0] and j != lb[1] and j != ub[1] and w(i, j) > 0:
                    X_Omega.append([i, j, t])
        for i in omega_X:
            for j in omega_Y:
                if w(i, j) < 0:
                    X_omega.append([i, j, t])
            # 单独再生成内边界网格点
            x1 = i
            delta = (1 - ((x1-G1)/a)**2)*b**2
            if delta >= 0:
                sqrt_delta = math.sqrt(delta)
                x2_1 = sqrt_delta + G2
                x2_2 = -sqrt_delta + G2
                if x2_1 == x2_2:
                    X_gamma.append([x1, x2_1, t])
                else:
                    X_gamma.append([x1, x2_1, t])
                    X_gamma.append([x1, x2_2, t])
    X_Omega = np.asarray(X_Omega, dtype=float)
    X_omega = np.asarray(X_omega, dtype=float)
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
        'psi0': psi0,
        'X_Omega': X_Omega,
        'X_omega': X_omega,
        'X_gamma': X_gamma,
        'delta_T': delta_T,
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
