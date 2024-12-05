import datetime
import logging
import math
import scipy.io
import os
import random
import sys
import time
import numpy as np
import torch
from parabolic.PINN_origin_time_batch2 import PINN

TASK_NAME = 'task_pinn_origin_batch2'
now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
root_path = '/home/dell/yangqh/FictitiousDomain/2D_Parabolic_moving_w/data/'
# root_path = 'c:/softwarefiles/VSCodeProjects/FictitiousDomain/2D_Parabolic_moving_w/data/'
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
    t_lb =[0]
    t_ub = [1]

    layers = [2, 40, 40, 40, 40, 1]
    t_layers = [1, 8, 40]
    param_dict = {
        'lb': lb,
        'ub': ub,
        't_lb': t_lb,
        't_ub': t_ub,
        'layers': layers,
        't_layers': t_layers,
        'device': device,
        'path': path,
        'root_path': root_path,
    }

    sin = torch.sin
    cos = torch.cos
    pi = torch.pi
    # 定义移动速度
    v1 = 1
    v2 = 1
    # 定义旋转速度
    theta = 10
    # delta_t = 1e-3

    # 椭圆区域参数
    G1 = 2
    G2 = 2
    a = 1/4
    b = 1/8
    def w(x1, x2, v1, v2, theta, t): return (((x1-G1-v1*t)*cos(theta*t)-(x2-G2-v2*t) *
                                              sin(theta*t))/a)**2+(((x1-G1-v1*t)*sin(theta*t)+(x2-G2-v2*t)*cos(theta*t))/b)**2 - 1

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

    data = scipy.io.loadmat(root_path + '/origin2.mat')
    X_Region = data['X_Region']
    X_gamma = data['X_gamma']
    T = data['T']
    

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
        'X_Region': X_Region,
        'X_gamma': X_gamma,
        'T': T,
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
