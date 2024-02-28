"""
@Name: create.py
@Auth: Huohuo
@Date: 2023/6/12-14:38
@Desc: 
@Ver : 
code_idea
"""
import ctypes

# 定义共享内存中的数据结构
class MyData(ctypes.Structure):
    _fields_ = [
        ('value1', ctypes.c_int),
        ('value2', ctypes.c_float),
        # 添加其他字段...
    ]

# 打开共享内存
shared_memory = ctypes.WinDLL('SharedMemory.dll')  # 替换为共享内存所在的动态链接库

# 连接到共享内存
shared_memory.connect()

# 读取共享内存中的数据
data = MyData()
shared_memory.read_data(ctypes.byref(data))

# 访问共享内存中的数据
value1 = data.value1
value2 = data.value2
# ...

# 关闭共享内存连接
shared_memory.disconnect()

