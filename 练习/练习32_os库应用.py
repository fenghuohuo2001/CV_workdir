# -*- 练习32 -*-
"""
功能：使用os库对文件夹内容进行操作
作者：fenghuohuo
日期：2021年11月15日
"""
import os

print(os.sep)   # 输出目录分割符
print(os.name)  # 输出当前操作系统内核 windows->NT内核
print('当前路径：', os.getcwd())  # 查看当前.py脚本工作目录

os.chdir("D:\\1.Desktop file\\picture")  # 修改当前路径为"D:\\1.Desktop file\\picture"
print('修改后路径：', os.getcwd())

os.getenv('PATH')       # 获取环境变量
os.environ()            # 获取并修改环境变量
os.mkdir("dirname")     # 在当前路径下建立一个子文件夹
os.rmdir("dirname")     # 删除一个文件夹
os.rename("my_file.txt", "cc.doc")      # 修改路径下文件的名字
os.remove("my_file.txt")    # 删除文件

os.listdir()            # 列出目录下所有文件

os.path.abspath(path)	    # 返回path在当前系统中的绝对路径
os.path.normpath(path)	    # 归一化path的表示形式(统一用\\分割路径)
os.path.relpath(path)	    # 返回当前程序与文件之前的相对路径
os.path.dirname(path)	    # 返回path中的目录路径
os.path.basename(path)	    # 返回path中最后的文件路径
os.path.join(path,*paths)	# 组合path和paths，返回一个字符串
os.path.exists(path)	    # 判断path对应文件或目录是否存在，返回布尔类型
os.path.isfile(path)	    # 判断path所对应的是否是已存在的文件，返回布尔类型
os.path.isdir(path)	        # 判断path所对应的是否是已存在的目录，返回布尔类型
os.path.getatime(path)	    # 返回path对应文件或目录上一次访问的时间(access)
os.path.getmtime(path)	    # 返回path对应文件或目录上一次修改的时间(modify)
os.path.getctime(path)	    # 返回path对应文件或目录创建的时间(create)
os.path.getsize(path)	    # 返回path对应文件的大小，以字节为单位

