"""
@Name: kalman_filter.py
@Auth: Huohuo
@Date: 2023/6/12-16:02
@Desc: 
@Ver : 
code_idea
https://blog.csdn.net/Canvaskan/article/details/115503937
"""
import numpy as np
import matplotlib.pyplot as plt

# 一步预测
def kf_predict(x0, p0, A, Q, B, u1):
    x10 = np.dot(A, x0) + np.dot(B, u1)
    p10 = np.dot(np.dot(A, p0), A.T) + Q
    return (x10, p10)

# 测量更新
def kf_update(x10, p10, Z, H, R):
    K = np.dot(np.dot(p10, H.T), np.linalg.pinv(np.dot(np.dot(H,p10), H.T) + R))
    x1 = x10 + np.dot(K, Z - np.dot(H, x10))
    p1 = np.dot(np.eye(K.shape[0]) - np.dot(K, H), p10)
    return (x1, p1, K)


"""
加速度白噪声建模
状态方程：
x' = v'
v' = a'
a' = 0 
离散化得到；
x(k) = x(k-1)+t*v(k)+0.5*t^2*a(k)
v(k) = v(k-1)+t*a(k)
a(k) = a(k-1)

观测方程：
z(k) = x(k) + e
"""

n = 20  # 数据量
nx = 3  # 变量数量
t = np.linspace(0, 3, n)  # 时间序列
dt = t[1] - t[0]

# 真实函数关系
a_true = np.ones(n) * 9.8 + np.random.normal(0, 1, size=n)
v_true = np.cumsum(a_true * dt)
x_true = np.cumsum(v_true * dt)
X_true = np.concatenate([x_true, v_true, a_true]).reshape([nx, -1])

# 观测噪声协方差！！！！！！！！！！！！！！！！！！！！（可调整）
R = np.diag([10 ** 2])

# 仿真观测值
e = np.random.normal(0, 2, n)
x_obs = x_true + e

# 计算系数
A = np.array([1, dt, 0.5 * dt ** 2,
              0, 1, dt,
              0, 0, 1]).reshape([nx, nx])
B = 0
U1 = 0

# 状态假设（观测）初始值
x0 = 0
v0 = 0
a0 = 10.0
X0 = np.array([x0, v0, a0]).reshape(nx, 1)

# 初始状态不确定度！！！！！！！！！！！！！！！！（可调整）
P0 = np.diag([0 ** 2, 0 ** 2, 0.2 ** 2])

# 状态递推噪声协方差！！！！！！！！！！！！！！！！！！（可调整）
Q = np.diag([0.0 ** 2, 0 ** 2, 1 ** 2])

###开始处理
X1_np = np.copy(X0)
P1_list = [P0]
X10_np = np.copy(X0)
P10_list = [P0]

for i in range(n):
    Zi = np.array(x_obs[i]).reshape([1, 1])
    Hi = np.array([1, 0, 0]).reshape([1, nx])

    if (i == 0):
        continue
    else:
        Xi = X1_np[:, i - 1].reshape([nx, 1])
        Pi = P1_list[i - 1]
        X10, P10 = kf_predict(Xi, Pi, A, Q, B, U1)

        X10_np = np.concatenate([X10_np, X10], axis=1)
        P10_list.append(P10)

        X1, P1, K = kf_update(X10, P10, Zi, Hi, R)
        X1_np = np.concatenate([X1_np, X1], axis=1)
        P1_list.append(P1)

# 结束，绘图
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x_true, 'k-', label="Truth")
ax1.plot(X1_np[0, :], 'go--', label="Kalman Filter")
ax1.scatter(np.arange(n), x_obs, label="Observation", marker='*')

plt.legend()
plt.show()
