# -*- 练习1-5 -*-
"""
功能：输入输出格式
作者：fenghuohuo
日期：2021年11月15日
"""


x = input("请输入x=")
y = input("请输入y=")
z = x+y
print("x+y="+z)

for x in range(1, 11):
  print(str(x).rjust(2), str(x*x).rjust(3), end=' ')
  print(str(x*x*x).rjust(4))


print("Python is really a great language,", "isn't it?")

li = ['hoho',18]

print('my name is {} ,age {}'.format('hoho',18))

closed = 1
print ("Closed or not : ", closed)