"""
@Name: kalman_filter-v2.py
@Auth: Huohuo
@Date: 2023/6/16-9:39
@Desc: 
@Ver : 
code_idea

真实值： x=-0.377
A=1
H=1

"""
import numpy as np
import pylab
import cv2

# -------------------------------------------
#           假设A=1， H=1
#             参数初始化
# -------------------------------------------
n_iter = 50             # 估计次数
sz = (n_iter, )         # size of array
x = -0.37728            # truth
z = np.random.normal(x, 0.1, size=sz)   # 观测值 ,观测时存在噪声

Q = 1e-5                # 过程激励噪声协方差 process variance

# -------------------------------------------
#               初始化变量
# -------------------------------------------
xhat = np.zeros(sz)     # x滤波估计值
P = np.zeros(sz)        # 滤波估计协方差矩阵
xhatminus = np.zeros(sz)    # x估计值
Pminus = np.zeros(sz)   # 估计协方差矩阵
K = np.zeros(sz)        # kalman filter增益

R = 0.1 ** 2            # 测量噪声协方差 estimate of measurement variance, change to see effect

# 初始估计
xhat[0] = 0.0
P[0] = 1.0              # 假设完全正相关，变量同步变化

# -------------------------------------------
#               预测 + 更新
#       从第2个开始，因为第1个已被初始值占用
# -------------------------------------------
for k in range(1, n_iter):
    print(f"--------------第{k}轮预测------------------", )
    # -----------预测------------
    xhatminus[k] = xhat[k-1]            # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
    Pminus[k] = P[k-1] + Q              # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1
    print(f"xhatminus[{k}] is", xhatminus[k])
    print(f"Pminus[{k}] is", Pminus[k])

    # -----------更新------------
    K[k] = Pminus[k] / (Pminus[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
    xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])      # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
    P[k] = (1 - K[k]) * Pminus[k]       # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1
    print(f"K[{k}] is", K[k])
    print(f"xhat[{k}] is", xhat[k])
    print(f"P[{k}] is", P[k])


pylab.figure()
pylab.plot(z, 'k+', label='noisy measurements')  # 观测值
pylab.plot(xhat, 'b-', label='a posteri estimate')  # 滤波估计值
pylab.axhline(x, color='g', label='truth value')  # 真实值
pylab.legend()
pylab.xlabel('Iteration')
pylab.ylabel('Voltage')

pylab.figure()
valid_iter = range(1, n_iter)  # Pminus not valid at step 0
pylab.plot(valid_iter, Pminus[valid_iter], label='a priori error estimate')
pylab.xlabel('Iteration')
pylab.ylabel('$(Voltage)^2$')
pylab.setp(pylab.gca(), 'ylim', [0, .01])
pylab.show()





























