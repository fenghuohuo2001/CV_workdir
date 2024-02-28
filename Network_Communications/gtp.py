import socket

# 创建 socket 对象
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 获取本地主机名
host = socket.gethostname()
print(host)
port = 12345

# 绑定端口
server_socket.bind((host, port))

# 设置最大连接数，超过后排队
server_socket.listen(5)

while True:
    # 建立客户端连接
    client_socket, addr = server_socket.accept()
    print('连接地址：', addr)

    # 接收数据
    data = client_socket.recv(1024)
    if not data:
        break

    print('接收到的数据：', data.decode('utf-8'))

    # 发送响应数据
    response = '你好，我已收到你的消息'
    client_socket.send(response.encode('utf-8'))

    client_socket.close()
