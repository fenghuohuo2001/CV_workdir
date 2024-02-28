"""
@Name: 1.切片.py
@Auth: Huohuo
@Date: 2023/2/22-15:37
@Desc: 
@Ver : 
code_idea
"""
import numpy as np

x = np.arange(6).reshape(2, 3)
x = np.matrix(x)

print("x is \n", x, '\n')


print("x[1, :] is \n", x[1, :], '\n')
print("x[1] is \n", x[1], '\n')

print("x[0][0] is \n", x[0][0], '\n')
print("x[1, 1] is \n", x[1, 1], '\n')




