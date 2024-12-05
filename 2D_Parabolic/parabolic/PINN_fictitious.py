import logging
import os
import time
import numpy as np
import torch
from torch import autograd, nn
from torch.autograd import Variable


class PINN(nn.Module):
    def __init__(self, param_dict, train_dict):
        super(PINN, self).__init__()
        # 设置使用设备:cpu, cuda
        lb, ub, self.psi1_layers, self.psi2_layers, self.device, self.path, self.root_path = self.unzip_param_dict(
            param_dict=param_dict)
        self.G1, self.G2, self.a, self.b, self.w, self.alpha, self.mu, self.ls, self.f, self.g0, self.g1, self.psi0, X_Omega, X_omega, X_Gamma, X_gamma, X_init = self.unzip_train_dict(
            train_dict=train_dict)

        # 上下界
        self.lb = self.data_loader(lb, requires_grad=False)
        self.ub = self.data_loader(ub, requires_grad=False)

        # 训练数据：坐标
        self.X_Omega = self.data_loader(X_Omega)
        self.X_omega = self.data_loader(X_omega)
        self.X_Gamma = self.data_loader(X_Gamma)
        self.X_gamma = self.data_loader(X_gamma)
        self.X_init = self.data_loader(X_init)

        # 激活函数
        self.act_func = torch.tanh

        # 初始化网络参数
        self.psi1_weights, self.psi1_biases = self.initialize_NN(
            self.psi1_layers)
        self.psi2_weights, self.psi2_biases = self.initialize_NN(
            self.psi2_layers)
        self.params = self.psi1_weights + self.psi1_biases + \
            self.psi2_weights + self.psi2_biases

        # 训练用的参数
        self.optimizer = None
        self.optimizer_name = None
        self.start_time = None
        # 小于这个数是开始保存模型
        self.min_loss = 1e8
        self.loss = None
        # 记录运行步数
        self.nIter = 0
        # 损失计算方式
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        # self.loss_fn = torch.nn.MSELoss(reduction='sum')

    # 参数读取
    def unzip_param_dict(self, param_dict):
        param_data = (param_dict['lb'], param_dict['ub'],
                      param_dict['psi1_layers'],
                      param_dict['psi2_layers'],
                      param_dict['device'], param_dict['path'],
                      param_dict['root_path'])
        return param_data

    def unzip_train_dict(self, train_dict):
        train_data = (
            train_dict['G1'],
            train_dict['G2'],
            train_dict['a'],
            train_dict['b'],
            train_dict['w'],
            train_dict['alpha'],
            train_dict['mu'],
            train_dict['ls'],
            train_dict['f'],
            train_dict['g0'],
            train_dict['g1'],
            train_dict['psi0'],
            train_dict['X_Omega'],
            train_dict['X_omega'],
            train_dict['X_Gamma'],
            train_dict['X_gamma'],
            train_dict['X_init'],
        )
        return train_data

    # 初始化网络
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = Variable(torch.zeros([1, layers[l + 1]],
                                     dtype=torch.float32)).to(self.device)
            b.requires_grad_()
            weights.append(W)
            biases.append(b)
        return weights, biases

    # 网络前向执行
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            X = self.act_func(torch.add(torch.matmul(X, W), b))

        W = weights[-1]
        b = biases[-1]
        Y = torch.add(torch.matmul(X, W), b)  # .requires_grad_()
        return Y

    # 参数Xavier初始化
    def xavier_init(self, size):
        W = Variable(nn.init.xavier_normal_(torch.empty(size[0], size[1]))).to(
            self.device)
        W.requires_grad_()
        return W

    # 数据加载函数，转变函数类型及其使用设备设置
    def data_loader(self, x, requires_grad=True):
        x_tensor = torch.tensor(x,
                                requires_grad=requires_grad,
                                dtype=torch.float32)
        return x_tensor.to(self.device)

    # 左边归一化函数，[-1, 1], lb:下确界，ub:上确界
    def coor_shift(self, X, lb, ub):
        X_shift = 2.0 * (X - lb) / (ub - lb) - 1.0
        # X_shift = torch.from_numpy(X_shift).float().requires_grad_()
        return X_shift

    # 将数据从设备上取出
    def detach(self, data):
        return data.detach().cpu().numpy()

    def net_psi1(self, x1, x2, t):
        X = torch.cat((x1, x2, t), 1)
        X = self.coor_shift(X, self.lb, self.ub)
        N = self.neural_net(X, self.psi1_weights, self.psi1_biases)
        # g = (1 - torch.exp(-(x1-self.lb[0]))) * (1 - torch.exp(-(x1-self.ub[0])))*(
        #     1 - torch.exp(-(x2-self.lb[1]))) * (1 - torch.exp(-(x2-self.ub[1])))
        # 强制Dirichlet边界条件、不用强制初值条件
        g = (x1-self.lb[0])*(x1-self.ub[0])*(x2-self.lb[1])*(x2-self.ub[1])
        psi1 = g * N + self.g0(x1, x2, t)
        return psi1

    # psi2 没有强制边界条件
    def net_psi2(self, x1, x2, t):
        X = torch.cat((x1, x2, t), 1)
        X = self.coor_shift(X, self.lb, self.ub)
        # X = self.coor_shift(X, self.w_lb, self.w_ub)
        psi2 = self.neural_net(X, self.psi2_weights, self.psi2_biases)
        return psi2

    # 前向函数
    def forward(self, x1, x2, t):
        psi1 = self.net_psi1(x1, x2, t)
        # return self.detach(u), self.detach(lambda_)
        return psi1

    # 损失函数计算损失并返回
    def loss_func(self, pred_, true_=None):
        # 采用MSELoss
        if true_ is None:
            true_ = torch.zeros_like(pred_).to(self.device)
            # true_ = self.data_loader(true_)
        return self.loss_fn(pred_, true_)

    # 直接计算一阶导数
    def compute_grad(self, u, x):
        u_x = autograd.grad(u.sum(), x, create_graph=True)[0]
        return u_x

    # Omega区域损失
    def loss_Omega(self):
        x1 = self.X_Omega[:, 0:1]
        x2 = self.X_Omega[:, 1:2]
        t = self.X_Omega[:, 2:3]
        psi1 = self.net_psi1(x1, x2, t)
        psi1_t = self.compute_grad(psi1, t)
        psi1_x1 = self.compute_grad(psi1, x1)
        psi1_x2 = self.compute_grad(psi1, x2)
        psi1_x1_x1 = self.compute_grad(psi1_x1, x1)
        psi1_x2_x2 = self.compute_grad(psi1_x2, x2)

        equation = psi1_t - self.mu * \
            (psi1_x1_x1 + psi1_x2_x2) - self.f(x1, x2, t)
        return self.loss_func(equation)

    # omega区域损失
    def loss_omega(self):
        x1 = self.X_omega[:, 0:1]
        x2 = self.X_omega[:, 1:2]
        t = self.X_omega[:, 2:3]

        # 计算psi1
        psi1 = self.net_psi1(x1, x2, t)
        psi1_t = self.compute_grad(psi1, t)
        psi1_x1 = self.compute_grad(psi1, x1)
        psi1_x2 = self.compute_grad(psi1, x2)
        psi1_x1_x1 = self.compute_grad(psi1_x1, x1)
        psi1_x2_x2 = self.compute_grad(psi1_x2, x2)

        equation1 = psi1_t - self.mu * \
            (psi1_x1_x1 + psi1_x2_x2)

        # 计算psi2
        psi2 = self.net_psi2(x1, x2, t)
        psi2_t = self.compute_grad(psi2, t)
        psi2_x1 = self.compute_grad(psi2, x1)
        psi2_x2 = self.compute_grad(psi2, x2)
        psi2_x1_x1 = self.compute_grad(psi2_x1, x1)
        psi2_x2_x2 = self.compute_grad(psi2_x2, x2)

        equation2 = psi2_t - self.mu * \
            (psi2_x1_x1 + psi2_x2_x2)

        loss_omega = self.loss_func(equation1, equation2)

        # # 最小化J 用Monte-Carlo 求积分
        # loss_J = torch.pi/2 * self.a * self.b / x2.size(0) * (self.alpha * torch.sum(
        #     (psi2-psi1)**2) + self.mu*torch.sum((psi2_x1-psi1_x1)**2 + (psi2_x2-psi1_x2)**2))
        # # 改为MSE
        # loss_J = self.loss_func(psi2, psi1)

        return loss_omega

    # 外边界损失
    def loss_Gamma(self, use_Gamma=False):
        loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        loss.requires_grad_()
        if use_Gamma:
            x1 = self.X_Gamma[:, 0:1]
            x2 = self.X_Gamma[:, 1:2]
            t = self.X_Gamma[:, 2:3]
            psi = self.net_psi(x1, x2, t)
            equation = psi - self.g0(x1, x2, t)
            loss = loss + self.loss_func(equation)
        return loss

    # 初值损失
    def loss_init(self, use_init=False):
        loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        loss.requires_grad_()
        if use_init:
            x1 = self.X_init[:, 0:1]
            x2 = self.X_init[:, 1:2]
            t = self.X_init[:, 2:3]
            psi = self.net_psi(x1, x2, t)
            equation = psi - self.psi0(x1, x2)
            loss = loss + self.loss_func(equation)
        return loss

    # 内边界损失
    def loss_gamma(self):
        x1 = self.X_gamma[:, 0:1]
        x2 = self.X_gamma[:, 1:2]
        t = self.X_gamma[:, 2:3]
        # 计算psi1
        psi1 = self.net_psi1(x1, x2, t)
        # 计算psi2
        psi2 = self.net_psi2(x1, x2, t)
        # psi2_sign = -1 * torch.sign(psi2)
        psi2_x1 = self.compute_grad(psi2, x1)
        psi2_x2 = self.compute_grad(psi2, x2)
        # 求单位法向量(n1,n2)
        # psi2_sign = torch.sign(psi2_x1*(x1-self.G1) + psi2_x2*(x2-self.G2))
        # sqrt_ = torch.sqrt(psi2_x1**2 + psi2_x2**2)
        # n1 = psi2_x1 / sqrt_ * psi2_sign
        # n2 = psi2_x2 / sqrt_ * psi2_sign

        # 1 表示外法向 -1表示内法向
        sign_ = -1
        n1 = 2*(x1-self.G1)/self.a**2
        n2 = 2*(x2-self.G2)/self.b**2
        sqrt_ = torch.sqrt(n1**2 + n2**2)
        # g1 内法向
        n1 = n1 / sqrt_ * sign_
        n2 = n2 / sqrt_ * sign_
        g1 = self.g1(x1, x2, t, n1, n2)
        # psi2 外法向
        n1 = n1 * sign_
        n2 = n2 * sign_

        equation = self.mu*(n1*psi2_x1 + n2*psi2_x2 - psi1 /
                            self.ls) + g1

        return self.loss_func(equation)

    # 训练一次
    def optimize_one_epoch(self):
        if self.start_time is None:
            self.start_time = time.time()

        # 初始化loss为0
        self.optimizer.zero_grad()
        self.loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        self.loss.requires_grad_()

        loss_Omega = self.loss_Omega()
        loss_omega = self.loss_omega()
        loss_Gamma = self.loss_Gamma(use_Gamma=False)
        loss_gamma = self.loss_gamma()
        loss_init = self.loss_init(use_init=False)

        # 权重
        alpha_Omega = 1
        alpha_omega = 1
        alpha_Gamma = 1
        alpha_gamma = 1
        alpha_init = 1

        self.loss = loss_Omega * alpha_Omega + \
            loss_omega * alpha_omega +\
            loss_Gamma * alpha_Gamma + loss_gamma * alpha_gamma + loss_init * alpha_init
        # 反向传播
        self.loss.backward()
        # 运算次数加1
        self.nIter = self.nIter + 1

        # 保存模型
        loss = self.detach(self.loss)
        if loss < self.min_loss:
            self.min_loss = loss
            PINN.save(net=self,
                      path=self.root_path + '/' + self.path,
                      name='PINN')

        # 打印日志
        loss_remainder = 10
        if np.remainder(self.nIter, loss_remainder) == 0:
            # 打印常规loss
            loss_Omega = self.detach(loss_Omega)
            loss_omega = self.detach(loss_omega)
            loss_Gamma = self.detach(loss_Gamma)
            loss_gamma = self.detach(loss_gamma)
            loss_init = self.detach(loss_init)

            log_str = str(self.optimizer_name) + ' Iter ' + str(self.nIter) +\
                ' Loss ' + str(loss) + ' loss_Omega ' + str(loss_Omega) +\
                ' loss_omega ' + str(loss_omega) +\
                ' loss_Gamma ' + str(loss_Gamma) + ' loss_gamma ' + str(loss_gamma) + ' loss_init ' + str(loss_init) + ' LR ' +\
                str(self.optimizer.state_dict()[
                    'param_groups'][0]['lr']) + ' min_loss ' + str(self.min_loss)

            print(log_str)
            logging.info(log_str)

            # 打印耗时
            elapsed = time.time() - self.start_time
            print('Time: %.4fs Per %d Iterators' % (elapsed, loss_remainder))
            logging.info('Time: %.4f s Per %d Iterators' %
                         (elapsed, loss_remainder))
            self.start_time = time.time()
        return self.loss

    def train_Adam(self, optimizer, nIter, Adam_scheduler):
        self.optimizer = optimizer
        self.optimizer_name = 'Adam'
        self.scheduler = Adam_scheduler
        for it in range(nIter):
            self.optimize_one_epoch()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

    @ staticmethod
    def save(net, path, name='PINN'):
        if not os.path.exists(path):
            os.makedirs(path)
        # 保存神经网络
        torch.save(net, path + '/' + name + '.pkl')  # 保存整个神经网络的结构和模型参数
        # torch.save(net.state_dict(), name + '_params.pkl')  # 只保存神经网络的模型参数

    # 载入整个神经网络的结构及其模型参数
    @ staticmethod
    def reload_net(net_path):
        net = torch.load(net_path)
        return net
        # 只载入神经网络的模型参数，神经网络的结构需要与保存的神经网络相同的结构
