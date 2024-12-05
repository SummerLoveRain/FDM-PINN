import logging
import math
import os
from queue import Queue
import time
import numpy as np
import torch
from torch import autograd, nn
from torch.autograd import Variable


class PINN(nn.Module):
    # N_Dimmension 为基函数的个数设置
    def __init__(self, param_dict, train_dict):
        super(PINN, self).__init__()
        # 设置使用设备:cpu, cuda
        lb, ub, self.layers, self.device, self.path, self.root_path = self.unzip_param_dict(
            param_dict=param_dict)
        self.G1, self.G2, self.a, self.b, self.w, self.alpha, self.mu, self.ls, self.f, self.g0, self.g1, X_Region, X_gamma = self.unzip_train_dict(
            train_dict=train_dict)

        # 上下界
        self.lb = self.data_loader(lb, requires_grad=False)
        self.ub = self.data_loader(ub, requires_grad=False)

        # 训练数据：坐标
        self.X_Region = X_Region
        self.X_gamma = X_gamma

        # 记录每个点的有效范围
        self.X_Region_Range = []
        self.X_gamma_Range = []

        # 容忍度
        self.tolerance = 1e-4
        # 定义每个点的作用作用域范围
        self.region_delta_x = 1
        self.gamma_delta_x = 0.5
        # 定义队列，记录点
        QUEUE_MAX_SIZE = 1e3
        self.Region_Queue = Queue(QUEUE_MAX_SIZE)
        self.gamma_Queue = Queue(QUEUE_MAX_SIZE)

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

    def net_psi(self, x1, x2):
        X = torch.cat((x1, x2), 1)
        X = self.coor_shift(X, self.lb, self.ub)
        N = self.neural_net(X, self.weights, self.biases)
        # g = (1 - torch.exp(-(x1-self.lb[0]))) * (1 - torch.exp(-(x1-self.ub[0])))*(
        #     1 - torch.exp(-(x2-self.lb[1]))) * (1 - torch.exp(-(x2-self.ub[1])))
        g = (x1-self.lb[0])*(x1-self.ub[0])*(x2-self.lb[1])*(x2-self.ub[1])
        psi = g * N + self.g0(x1, x2)
        return psi

    # 前向函数
    def forward(self, x1, x2):
        psi = self.net_psi(x1, x2)
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
    def loss_region(self, X_Region, use_all=False):
        if use_all:
            X_Region = self.data_loader(self.X_Region)
            x1 = X_Region[:, 0:1]
            x2 = X_Region[:, 1:2]
        else:
            X_Region = self.data_loader([X_Region])
            x1 = X_Region[:, 0:1]
            x2 = X_Region[:, 1:2]
        psi = self.forward(x1, x2)
        psi_x1 = self.compute_grad(psi, x1)
        psi_x2 = self.compute_grad(psi, x2)
        psi_x1_x1 = self.compute_grad(psi_x1, x1)
        psi_x2_x2 = self.compute_grad(psi_x2, x2)

        equation = self.alpha * psi - self.mu * \
            (psi_x1_x1 + psi_x2_x2) - self.f(x1, x2)
        return self.loss_func(equation)

    # 内边界损失
    def loss_gamma(self, X_gamma, use_all=False):
        if use_all:
            X_gamma = self.data_loader(self.X_gamma)
            x1 = X_gamma[:, 0:1]
            x2 = X_gamma[:, 1:2]
        else:
            X_gamma = self.data_loader([X_gamma])
            x1 = X_gamma[:, 0:1]
            x2 = X_gamma[:, 1:2]
        psi = self.forward(x1, x2)
        # psi_sign = -1 * torch.sign(psi)
        psi_x1 = self.compute_grad(psi, x1)
        psi_x2 = self.compute_grad(psi, x2)
        # 求单位法向量(n1,n2)
        # psi_sign = -1 * \
        #     torch.sign(psi_x1*(x1-self.G1) + psi_x2*(x2-self.G2))
        # sqrt_ = torch.sqrt(psi_x1**2 + psi_x2**2)
        # n1 = psi_x1 / sqrt_ * psi_sign
        # n2 = psi_x2 / sqrt_ * psi_sign

        # 1 表示外法向 -1 表示内法向
        sign_ = -1
        n1 = 2*(x1-self.G1)/self.a**2
        n2 = 2*(x2-self.G2)/self.b**2
        sqrt_ = torch.sqrt(n1**2 + n2**2)
        n1 = n1 / sqrt_ * sign_
        n2 = n2 / sqrt_ * sign_

        equation = self.mu*(n1*psi_x1 + n2*psi_x2 + psi /
                            self.ls) - self.g1(x1, x2, n1, n2)
        return self.loss_func(equation)

    # 训练一次
    def optimize_one_epoch(self):
        if self.start_time is None:
            self.start_time = time.time()

        # 初始化loss为0
        self.optimizer.zero_grad()
        self.loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        self.loss.requires_grad_()

        # 当X_Region 和 X_gamma中的点都知道自己的范围后，且队列都为空时，则剩下的训练用全部的点来训练
        use_all = False
        if len(self.X_Region) != len(self.X_Region_Range) and self.Region_Queue.empty() and len(self.X_gamma) != len(self.X_gamma) and self.gamma_Queue.empty():
            use_all = True

        loss_region = self.loss_region(self.X_Region[-1], use_all=use_all)
        loss_gamma = self.loss_gamma(self.X_gamma[-1], use_all=use_all)

        # 权重
        alpha_region = 1
        alpha_gamma = 1

        self.loss = loss_region * alpha_region + loss_gamma * alpha_gamma
        # 反向传播
        self.loss.backward()
        # 运算次数加1
        self.nIter = self.nIter + 1

        # 保存模型
        loss = self.detach(self.loss)
        # 打印常规loss
        loss_region = self.detach(loss_region)
        loss_gamma = self.detach(loss_gamma)

        # 当损失足够小时，表示该区域点已经足够收敛了
        tmp_loss = loss_region
        if loss_region < self.tolerance and loss_gamma < self.tolerance:
            if len(self.X_Region) != len(self.X_Region_Range):
                X_Region = self.X_Region[-1]
                len_X = X_Region.size
                # 第一个位置记录当前迭代次数
                X_Region_Range = [self.nIter]
                # 判断当前点的各个方位，记录其有效范围
                # 每个坐标轴 应当由两个方位，即当前点的前后方位
                for i in range(len_X):
                    # 先判断正向坐标轴方向
                    region_delta_x = self.region_delta_x
                    last_X_Region = None
                    while True and region_delta_x > 0:
                        new_X_Region = [X_Region[j] for j in range(len_X)]
                        new_X_Region[i] = new_X_Region[i] + region_delta_x
                        # 先判断坐标是否在区域内
                        if self.lb[i] < new_X_Region[i] < self.ub[i] and self.w(new_X_Region[0], new_X_Region[1]) > 0:
                            # 当在探索范围内，则判断损失是否足够小
                            self.optimizer.zero_grad()
                            tmp_loss = self.detach(
                                self.loss_region(new_X_Region))
                            if tmp_loss < self.tolerance:
                                # 当损失足够小是，则将可能的定位点加入队列
                                if last_X_Region is not None:
                                    self.Region_Queue.put(last_X_Region)
                                # 记录该方向的有效范围
                                X_Region_Range.append(region_delta_x)
                                # 找到点后退出循环
                                break
                            else:
                                # 否则减小探索步长
                                region_delta_x = region_delta_x/2
                                # 记录上一个误差较大的点
                                last_X_Region = new_X_Region

                        else:
                            # 否则减小探索步长
                            region_delta_x = region_delta_x/2
                    # 再判断负向坐标轴方向
                    region_delta_x = self.region_delta_x
                    last_X_Region = None
                    while True and region_delta_x > 0:
                        new_X_Region = [X_Region[j] for j in range(len_X)]
                        new_X_Region[i] = new_X_Region[i] - region_delta_x
                        # 先判断坐标是否在区域内
                        if self.lb[i] < new_X_Region[i] < self.ub[i] and self.w(new_X_Region[0], new_X_Region[1]) > 0:
                            # 当在探索范围内，则判断损失是否足够小
                            self.optimizer.zero_grad()
                            tmp_loss = self.detach(
                                self.loss_region(new_X_Region))
                            if tmp_loss < self.tolerance:
                                # 当损失足够小是，则将可能的定位点加入队列
                                if last_X_Region is not None:
                                    self.Region_Queue.put(last_X_Region)
                                # 记录该方向的有效范围
                                X_Region_Range.append(region_delta_x)
                                # 找到点后退出循环
                                break
                            else:
                                # 否则减小探索步长
                                region_delta_x = region_delta_x/2
                                # 记录上一个误差较大的点
                                last_X_Region = new_X_Region

                        else:
                            # 否则减小探索步长
                            region_delta_x = region_delta_x/2
                self.X_Region_Range.append(X_Region_Range)

            # 当队列不为空
            if not self.Region_Queue.empty():
                X_Region = self.Region_Queue.get()
                X_Region = np.asarray(X_Region, dtype=float)
                self.X_Region = np.append(self.X_Region, [X_Region], axis=0)

            # 边界点也寻找下一个点
            # 当损失足够小时，表示该边界点已经足够收敛了
            if len(self.X_gamma) != len(self.X_gamma_Range):
                X_gamma = self.X_gamma[-1]
                len_X = X_gamma.size
                # 第一个位置记录当前迭代次数
                X_gamma_Range = [self.nIter]
                # 判断当前点的各个方位，记录其有效范围
                # 每个坐标轴 应当由两个方位，即当前点的前后方位
                # 椭圆只移动x轴的坐标

                # 先判断正向坐标轴方向
                i = 0
                gamma_delta_x = self.gamma_delta_x
                last_X_gamma = None
                while True and gamma_delta_x > 0:
                    new_X_gamma = [X_gamma[j] for j in range(len_X)]
                    new_X_gamma[i] = new_X_gamma[i] + gamma_delta_x
                    # 先判断坐标是否在区域内
                    if (self.G1-self.a) <= new_X_gamma[i] <= (self.G1+self.a):
                        # 计算y坐标
                        x1 = new_X_gamma[i]
                        delta = (1 - ((x1-self.G1)/self.a)**2)*self.b**2
                        if delta >= 0:
                            sqrt_delta = math.sqrt(delta)
                            x2_1 = sqrt_delta + self.G2
                            x2_2 = -sqrt_delta + self.G2
                            if x2_1 == x2_2:
                                new_X_gamma[i+1] = x2_1
                                # 当在探索范围内，则判断损失是否足够小
                                self.optimizer.zero_grad()
                                tmp_loss = self.detach(
                                    self.loss_gamma(new_X_gamma))
                                if tmp_loss < self.tolerance:
                                    # 当损失足够小是，则将可能的定位点加入队列
                                    if last_X_gamma is not None:
                                        self.gamma_Queue.put(last_X_gamma)
                                    # 记录该方向的有效范围
                                    X_gamma_Range.append(gamma_delta_x)
                                    # 找到点后退出循环
                                    break
                                else:
                                    # 否则减小探索步长
                                    gamma_delta_x = gamma_delta_x/2
                                    # 记录上一个误差较大的点
                                    last_X_gamma = new_X_gamma
                            else:
                                new_X_gamma[i+1] = x2_1
                                # 当在探索范围内，则判断损失是否足够小
                                self.optimizer.zero_grad()
                                tmp_loss = self.detach(
                                    self.loss_gamma(new_X_gamma))
                                if tmp_loss < self.tolerance:
                                    # 当损失足够小是，则将可能的定位点加入队列
                                    if last_X_gamma is not None:
                                        self.gamma_Queue.put(last_X_gamma)
                                    # 记录该方向的有效范围
                                    X_gamma_Range.append(gamma_delta_x)
                                    # 找到点后退出循环
                                    break
                                else:
                                    # 否则减小探索步长
                                    gamma_delta_x = gamma_delta_x/2
                                    # 记录上一个误差较大的点
                                    last_X_gamma = new_X_gamma
                                new_X_gamma[i+1] = x2_2
                                # 当在探索范围内，则判断损失是否足够小
                                self.optimizer.zero_grad()
                                tmp_loss = self.detach(
                                    self.loss_gamma(new_X_gamma))
                                if tmp_loss < self.tolerance:
                                    # 当损失足够小是，则将可能的定位点加入队列
                                    if last_X_gamma is not None:
                                        self.gamma_Queue.put(last_X_gamma)
                                    # 记录该方向的有效范围
                                    X_gamma_Range.append(gamma_delta_x)
                                    # 找到点后退出循环
                                    break
                                else:
                                    # 否则减小探索步长
                                    gamma_delta_x = gamma_delta_x/2
                                    # 记录上一个误差较大的点
                                    last_X_gamma = new_X_gamma
                        else:
                            # 否则减小探索步长
                            gamma_delta_x = gamma_delta_x/2
                    else:
                        # 否则减小探索步长
                        gamma_delta_x = gamma_delta_x/2
                # 再判断负向坐标轴方向
                gamma_delta_x = self.gamma_delta_x
                last_X_gamma = None
                while True and gamma_delta_x > 0:
                    new_X_gamma = [X_gamma[j] for j in range(len_X)]
                    new_X_gamma[i] = new_X_gamma[i] - gamma_delta_x
                    # 先判断坐标是否在区域内
                    if (self.G1-self.a) <= new_X_gamma[i] <= (self.G1+self.a):
                        # 计算y坐标
                        x1 = new_X_gamma[i]
                        delta = (1 - ((x1-self.G1)/self.a)**2)*self.b**2
                        if delta >= 0:
                            sqrt_delta = math.sqrt(delta)
                            x2_1 = sqrt_delta + self.G2
                            x2_2 = -sqrt_delta + self.G2
                            if x2_1 == x2_2:
                                new_X_gamma[i+1] = x2_1
                                # 当在探索范围内，则判断损失是否足够小
                                self.optimizer.zero_grad()
                                tmp_loss = self.detach(
                                    self.loss_gamma(new_X_gamma))
                                if tmp_loss < self.tolerance:
                                    # 当损失足够小是，则将可能的定位点加入队列
                                    if last_X_gamma is not None:
                                        self.gamma_Queue.put(last_X_gamma)
                                    # 记录该方向的有效范围
                                    X_gamma_Range.append(gamma_delta_x)
                                    # 找到点后退出循环
                                    break
                                else:
                                    # 否则减小探索步长
                                    gamma_delta_x = gamma_delta_x/2
                                    # 记录上一个误差较大的点
                                    last_X_gamma = new_X_gamma
                            else:
                                new_X_gamma[i+1] = x2_1
                                # 当在探索范围内，则判断损失是否足够小
                                self.optimizer.zero_grad()
                                tmp_loss = self.detach(
                                    self.loss_gamma(new_X_gamma))
                                if tmp_loss < self.tolerance:
                                    # 当损失足够小是，则将可能的定位点加入队列
                                    if last_X_gamma is not None:
                                        self.gamma_Queue.put(last_X_gamma)
                                    # 记录该方向的有效范围
                                    X_gamma_Range.append(gamma_delta_x)
                                    # 找到点后退出循环
                                    break
                                else:
                                    # 否则减小探索步长
                                    gamma_delta_x = gamma_delta_x/2
                                    # 记录上一个误差较大的点
                                    last_X_gamma = new_X_gamma
                                new_X_gamma[i+1] = x2_2
                                # 当在探索范围内，则判断损失是否足够小
                                self.optimizer.zero_grad()
                                tmp_loss = self.detach(
                                    self.loss_gamma(new_X_gamma))
                                if tmp_loss < self.tolerance:
                                    # 当损失足够小是，则将可能的定位点加入队列
                                    if last_X_gamma is not None:
                                        self.gamma_Queue.put(last_X_gamma)
                                    # 记录该方向的有效范围
                                    X_gamma_Range.append(gamma_delta_x)
                                    # 找到点后退出循环
                                    break
                                else:
                                    # 否则减小探索步长
                                    gamma_delta_x = gamma_delta_x/2
                                    # 记录上一个误差较大的点
                                    last_X_gamma = new_X_gamma
                        else:
                            # 否则减小探索步长
                            gamma_delta_x = gamma_delta_x/2
                    else:
                        # 否则减小探索步长
                        gamma_delta_x = gamma_delta_x/2
                self.X_gamma_Range.append(X_gamma_Range)

            # 当队列不为空
            if not self.gamma_Queue.empty():
                X_gamma = self.gamma_Queue.get()
                X_gamma = np.asarray(X_gamma, dtype=float)
                self.X_gamma = np.append(self.X_gamma, [X_gamma], axis=0)

        if use_all and loss < self.min_loss:
            self.min_loss = loss
            PINN.save(net=self,
                      path=self.root_path + '/' + self.path,
                      name='PINN')

        # 打印日志
        loss_remainder = 10
        if np.remainder(self.nIter, loss_remainder) == 0:

            log_str = str(self.optimizer_name) + ' Iter ' + str(self.nIter) + ' use_all ' + str(use_all) + ' Loss ' +\
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
