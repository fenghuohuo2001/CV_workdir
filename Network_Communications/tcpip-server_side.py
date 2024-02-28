"""
@Name: tcpip.py
@Auth: Huohuo
@Date: 2023/8/28-10:38
@Desc: 
@Ver : 
code_idea
"""
# ------------------------------------------------------
#                   tcp服务器构建流程
# ------------------------------------------------------
import socket

# ------------------------------------------------------
#                  1-创建socket对象
# ------------------------------------------------------
tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# ------------------------------------------------------
#                  2-标定本地ip与port
# ------------------------------------------------------
# 获取本地主机名
host = socket.gethostname()
print(host)
address = ('192.168.60.10', 12345)
# address = ('', 12345)
print(address)
tcp_server_socket.bind(address)

# ------------------------------------------------------
#                  3-listen使socket变为可被动链接
# ‘128’ 用于指定服务器套接字可以排队等待的最大连接数
# 使用socket创建的套接字默认的属性是主动的，使用listen将其变为被动，这样就可以接收别人的链接
# ------------------------------------------------------
tcp_server_socket.listen(128)

# ------------------------------------------------------
#                  4-等待客户端链接服务器
# 若有新的客户端来链接服务器，就生成一个新的socket专门为这个客户端服务
# client_socket：用来为这个客户端服务
# tcp_server_socket：就可以省下来专门等待其他新客户端的链接
# ------------------------------------------------------
client_socket, clientAddr = tcp_server_socket.accept()

# ------------------------------------------------------
#                  5-接收数据
# ------------------------------------------------------
recv_data = client_socket.recv(1024)        # 接收1024个字节
print("接收到的数据为：", recv_data.decode('utf-8'))

client_socket.close()

