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
        lb, ub, self.layers, self.device, self.path, self.root_path = self.unzip_param_dict(
            param_dict=param_dict)
        self.G1, self.G2, self.a, self.b, self.w, self.alpha, self.mu, self.ls, self.f, self.g0, self.g1, self.psi0, X_Region, X_gamma = self.unzip_train_dict(
            train_dict=train_dict)

        # 上下界
        self.lb = self.data_loader(lb, requires_grad=False)
        self.ub = self.data_loader(ub, requires_grad=False)

        # 训练数据：坐标 加载到cuda中
        self.X_Region = X_Region
        self.X_gamma = X_gamma

        # 激活函数
        self.act_func = torch.tanh

        # 初始化网络参数
        self.weights, self.biases = self.initialize_NN(self.layers)
        self.params = self.weights + self.biases
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
        param_data = (param_dict['lb'], param_dict['ub'], param_dict['layers'],
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
            train_dict['X_Region'],
            train_dict['X_gamma'],
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

    def net_psi(self, x1, x2, t):
        X = torch.cat((x1, x2, t), 1)
        X = self.coor_shift(X, self.lb, self.ub)
        N = self.neural_net(X, self.weights, self.biases)
        # g = (1 - torch.exp(-(x1-self.lb[0]))) * (1 - torch.exp(-(x1-self.ub[0])))*(
        #     1 - torch.exp(-(x2-self.lb[1]))) * (1 - torch.exp(-(x2-self.ub[1])))
        # 强制Dirichlet边界条件、不用强制初值条件
        g = (x1-self.lb[0])*(x1-self.ub[0])*(x2-self.lb[1])*(x2-self.ub[1])*t
        psi = g * N + self.g0(x1, x2, t)
        return psi

    # 前向函数
    def forward(self, x1, x2, t):
        psi = self.net_psi(x1, x2, t)
        # return self.detach(u), self.detach(lambda_)
        return psi

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

    # 区域损失
    def loss_region(self, X_Region):
        x1 = X_Region[:, 0:1]
        x2 = X_Region[:, 1:2]
        t = X_Region[:, 2:3]
        psi = self.net_psi(x1, x2, t)
        psi_t = self.compute_grad(psi, t)
        psi_x1 = self.compute_grad(psi, x1)
        psi_x2 = self.compute_grad(psi, x2)
        psi_x1_x1 = self.compute_grad(psi_x1, x1)
        psi_x2_x2 = self.compute_grad(psi_x2, x2)

        equation = psi_t - self.mu * \
            (psi_x1_x1 + psi_x2_x2) - self.f(x1, x2, t)
        return self.loss_func(equation)

    # 内边界损失
    def loss_gamma(self, X_gamma):
        x1 = X_gamma[:, 0:1]
        x2 = X_gamma[:, 1:2]
        t = X_gamma[:, 2:3]
        psi = self.net_psi(x1, x2, t)
        psi_x1 = self.compute_grad(psi, x1)
        psi_x2 = self.compute_grad(psi, x2)

        # 1 表示外法向 -1 表示内法向
        sign_ = -1
        n1 = 2*(x1-self.G1)/self.a**2
        n2 = 2*(x2-self.G2)/self.b**2
        sqrt_ = torch.sqrt(n1**2 + n2**2)
        n1 = n1 / sqrt_ * sign_
        n2 = n2 / sqrt_ * sign_

        equation = self.mu*(n1*psi_x1 + n2*psi_x2 + psi /
                            self.ls) - self.g1(x1, x2, t, n1, n2)
        return self.loss_func(equation)

    # 训练一次
    def optimize_one_epoch(self):
        if self.start_time is None:
            self.start_time = time.time()

        # 打乱数据
        X_Region = np.random.permutation(self.X_Region)
        X_gamma = np.random.permutation(self.X_gamma)
        len_X_Region = X_Region.shape[0]
        len_X_gamma = X_gamma.shape[0]
        # Region_Batch_Size = 13044
        # gamma_Batch_Size = 40
        Region_Batch_Size = len_X_Region//10
        gamma_Batch_Size = len_X_gamma//10
        iter = 0
        last_X_Region = None
        last_X_gamma = None
        while iter*Region_Batch_Size < len_X_Region or iter*gamma_Batch_Size < len_X_gamma:
            target_X_Region = None
            target_X_gamma = None
            if iter*Region_Batch_Size >= len_X_Region:
                target_X_Region = last_X_Region
            else:
                target_X_Region = X_Region[iter *
                                           Region_Batch_Size:(iter + 1) * Region_Batch_Size, :]
                last_X_Region = target_X_Region
            if iter*gamma_Batch_Size >= len_X_gamma:
                target_X_gamma = last_X_gamma
            else:
                target_X_gamma = X_gamma[iter *
                                         gamma_Batch_Size:(iter + 1) * gamma_Batch_Size, :]
                last_X_gamma = target_X_gamma
            # 成功获取数据后分批次训练
            if target_X_Region is not None and target_X_gamma is not None:
                X_R = self.data_loader(target_X_Region)
                X_g = self.data_loader(target_X_gamma)
                # 初始化loss为0
                self.optimizer.zero_grad()
                self.loss = torch.tensor(
                    0.0, dtype=torch.float32).to(self.device)
                self.loss.requires_grad_()
                loss_region = self.loss_region(X_R)
                loss_gamma = self.loss_gamma(X_g)

                # 权重
                alpha_region = 1
                alpha_gamma = 1

                self.loss = loss_region * alpha_region + loss_gamma * alpha_gamma
                # 反向传播
                self.loss.backward()

                loss = self.detach(self.loss)

                # 保存模型
                if loss < self.min_loss:
                    self.min_loss = loss
                    PINN.save(net=self,
                              path=self.root_path + '/' + self.path,
                              name='PINN')
                # 运算次数加1
                iter = iter + 1

        # 运算次数加1
        self.nIter = self.nIter + 1

        # 打印日志
        loss_remainder = 1
        if np.remainder(self.nIter, loss_remainder) == 0:
            # 打印常规loss
            loss_region = self.detach(loss_region)
            loss_gamma = self.detach(loss_gamma)

            log_str = str(self.optimizer_name) + ' Iter ' + str(self.nIter) + ' Loss ' +\
                str(loss) + ' loss_region ' + str(loss_region) +\
                ' loss_gamma ' + str(loss_gamma) + ' LR ' +\
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
