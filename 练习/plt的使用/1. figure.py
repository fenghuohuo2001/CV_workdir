# -*- 练习 -*-
"""
功能：
作者：fenghuohuo
日期：2021年11月29日
"""
# D:\anaconda\envs\pytorch\Lib\site-packages\PyQt5\Qt5\plugins
# bug记录，如果qt出现bug，找不到库，就直接pip uninstall pyqt5  +  pip install pyqt5

import matplotlib.pyplot as plt

# # 创建自定义图像
#
# fig = plt.figure(figsize=(4, 3), facecolor='blue')
# plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
plt.show()