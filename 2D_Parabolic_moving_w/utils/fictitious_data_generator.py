import math
import numpy as np
import scipy.io
import torch

# 设置定义域
lb = np.array([0, 0, 0])
ub = np.array([4, 4, 1])

sin = np.sin
cos = np.cos
pi = np.pi
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

# w用来判断坐标是否还在椭圆内


def w(x1, x2, t): return (((x1-G1-v1*t)*cos(2*pi-theta*t)-(x2-G2-v2*t) *
                           sin(2*pi-theta*t))/a)**2+(((x1-G1-v1*t)*sin(2*pi-theta*t)+(x2-G2-v2*t)*cos(2*pi-theta*t))/b)**2 - 1


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
            if i != lb[0] and i != ub[0] and j != lb[1] and j != ub[1] and w(i, j, t) > 0:
                X_Omega.append([i, j, t])
            # elif np_w(i, j, v1, v2, theta, t) < 0:
            #     X_omega.append([i, j, t])

# 先生成初始椭圆内部点，然后每次变换后修改
# tmp_Omega = []
tmp_omega = []
for i in omega_X:
    for j in omega_Y:
        if w(i, j, 0) < 0:
            tmp_omega.append([i, j])
# for i in Omega_X:
#     for j in Omega_Y:
#         if i!=lb[0] and i!=ub[0] and j!=lb[1] and j!=ub[1] and w(i, j, v1, v2, theta, 0) > 0:
#             tmp_Omega.append([i, j])
for t in T:
    if t == 0:
        continue
    v1_t = v1 * t
    v2_t = v2 * t
    theta_t = theta * t
    # 计算每个时间步的位移+旋转后的坐标
    for x1, x2 in tmp_omega:
        # 先移动回原点、然后旋转
        new_x1 = (x1-G1)*cos(theta_t)-(x2-G2)*sin(theta_t)
        new_x2 = (x1-G1)*sin(theta_t)+(x2-G2)*cos(theta_t)
        # 再移动
        new_x1 = new_x1 + G1 + v1_t
        new_x2 = new_x2 + G2 + v2_t

        print('x1:'+str(new_x1) + '\tx2:'+str(new_x2) +
              '\terror:'+str(w(new_x1, new_x2, t)))

        X_omega.append([new_x1, new_x2, t])
    # for x1, x2 in tmp_Omega:
    #     v1_t = v1 * t
    #     v2_t = v2 * t
    #     theta_t = theta * t
    #     new_x1 = (x1+v1_t)*cos(theta_t)-(x2+v2_t)*sin(theta_t)
    #     new_x2 = (x1+v1_t)*sin(theta_t)+(x2+v2_t)*cos(theta_t)
    #     X_Omega.append([new_x1, new_x2, t])

# 先生成初始椭圆边界点，然后每次变换后修改
tmp_gamma = []
for i in omega_X:
    x1 = i
    delta = (1 - ((x1-G1)/a)**2)*b**2
    if delta >= 0:
        sqrt_delta = math.sqrt(delta)
        x2_1 = sqrt_delta + G2
        x2_2 = -sqrt_delta + G2
        if x2_1 == x2_2:
            tmp_gamma.append([x1, x2_1])
        else:
            tmp_gamma.append([x1, x2_1])
            tmp_gamma.append([x1, x2_2])

# 单独再生成内边界网格点
for t in T:
    if t == 0:
        continue
    v1_t = v1 * t
    v2_t = v2 * t
    theta_t = theta * t
    # 计算每个时间表的位移+旋转后的坐标
    for x1, x2 in tmp_gamma:
        # new_x1 = (x1+v1_t)*cos(theta_t)-(x2+v2_t)*sin(theta_t)
        # new_x2 = (x1+v1_t)*sin(theta_t)+(x2+v2_t)*cos(theta_t)

        # 先移动回原点、然后旋转
        new_x1 = (x1-G1)*cos(theta_t)-(x2-G2)*sin(theta_t)
        new_x2 = (x1-G1)*sin(theta_t)+(x2-G2)*cos(theta_t)
        # 再移动
        new_x1 = new_x1 + G1 + v1_t
        new_x2 = new_x2 + G2 + v2_t

        print('x1:'+str(new_x1) + '\tx2:'+str(new_x2) +
              '\terror:'+str(w(new_x1, new_x2, t)))

        X_gamma.append([new_x1, new_x2, t])

X_Omega = np.asarray(X_Omega, dtype=float)
X_omega = np.asarray(X_omega, dtype=float)
X_gamma = np.asarray(X_gamma, dtype=float)


root_path = '/home/dell/yangqh/FictitiousDomain/2D_Parabolic_moving_w/data/'

scipy.io.savemat(root_path + '/fictitious'+'_'+str(Omega_GRID_SIZE) + '_' + str(omega_GRID_SIZE)+'_' + str(delta_T)+'.mat',
                 {'X_Omega': X_Omega, 'X_omega': X_omega, 'X_gamma': X_gamma})
