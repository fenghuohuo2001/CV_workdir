"""
@Name: define_struct.py
@Auth: Huohuo
@Date: 2023/6/16-13:57
@Desc: 
@Ver : 
code_idea
"""
import ctypes

# 定义结构体
class _SmartADASExchangeInfo(ctypes.Structure):
    # 上传给stm32的数据
    _fields_ = [
        ("temperature", ctypes.c_ushort),        # 温度20
        ("humidity", ctypes.c_ushort),           # 湿度80
        ("BDoor", ctypes.c_ushort, 1),                   # 后炮0
        ("RBDDoor", ctypes.c_ushort, 1),                 # 右后下0
        ("RBUDoor", ctypes.c_ushort, 1),                 # 右后上
        ("RFBDoor", ctypes.c_ushort, 1),                 # 右前
        ("LBDDoor", ctypes.c_ushort, 1),                 # 左后下
        ("LBUoor", ctypes.c_ushort, 1),                  # 左上
        ("LFDoor", ctypes.c_ushort, 1),                  # 左前
        ("FDoor", ctypes.c_ushort, 1),                   # 前炮
        # ("WaterDev", ctypes.c_ushort, 2),                  # 水炮阀门
        ("ManAdjust", ctypes.c_ushort, 2),                   # 手动调节标识,置为1，发送0x02数据包  0 1
        ("mLeftCamera", ctypes.c_ushort, 2),                 # 左侧摄像头状态给
        ("mRightCamera", ctypes.c_ushort, 2),                # 右侧侧摄像头状态给
        ("CameraSwitch", ctypes.c_ushort, 2),                # 1-后置摄像头工作 2-前置摄像头的车道线识别工作 3-前置摄像头植物检测工作

        ("WaterLevel", ctypes.c_ushort),                   # 温度
        ("PressLevel", ctypes.c_ushort),                   # 压力

        ("electronDev", ctypes.c_ulong, 8),                   # 电磁阀开度 0%~100%对应0x00~0x64
        ("MainflowVal", ctypes.c_ulong, 8),                   # 换算成对应的十进制/100=实际值单位L/s
        ("SecflowVal", ctypes.c_ulong, 8),                    # 回流管道流量2：换算成对应的十进制/100=实际值单位L/s
        ("pitchAng", ctypes.c_ulong, 8),                      # 水炮俯仰角度：0~180度
        ("horizAng", ctypes.c_ulong, 8),                      # 水炮左右转动角度：0~180度
        ("LeftPerson", ctypes.c_ulong, 2),                    # 左侧侧有行人 0  1
        ("RightPerson", ctypes.c_ulong, 2),                   # 右侧有行人
        ("RoadLine", ctypes.c_ulong, 8),                      # 当前车道数：4车道0x04，8车道0x08(后续可能要改成有符号的类型)
        ("horizDis", ctypes.c_ulong, 8),                      # 植物与摄像头之间的水平距离
        ("verticalDis", ctypes.c_ulong, 8),                   # 植物与摄像头之间的垂直距离
        ("DevLen", ctypes.c_ulong, 4),                        # 炮口伸长0/4对应0x00、1/4对应0x01、2/4对应0x02、3/4对应0x03、4/4对应0x04
        ("LastLeftPerson", ctypes.c_ulong, 2),                # 上一次左侧侧有行人 0
        ("LastRightPerson", ctypes.c_ulong, 2),               # 上一次右侧有行人
        ("LastRoadLine", ctypes.c_ulong, 4),                  # 上一次当前车道数：4车道0x04，8车道0x08
        ("LasthorizDis", ctypes.c_ulong, 8),                  # 上一次植物与摄像头之间的水平距离
        ("LastverticalDis", ctypes.c_ulong, 8),               # 上一次植物与摄像头之间的垂直距离
        ("LastDevLen", ctypes.c_ulong, 8),                    # 上一次炮口伸长0/4对应0x00、1/4对应0x01、2/4对应0x02、3/4对应0x03、4/4对应0x04

        ("RPersonWarn", ctypes.c_ulong, 8),                  # 右侧行人报警
        ("LPersonWarn", ctypes.c_ulong, 8),                  # 左侧行人报警

        ("showDsm", ctypes.c_ulong, 8),                      # 显示模式 0xA1-DSM 0xA2-后置 0XA3 左侧 0xA4 右侧
        ("PicDataValid", ctypes.c_ulong, 8),                 # 视频帧是否正常
        ("rows", ctypes.c_ulong, 16),                        # 视频高

        ("cols", ctypes.c_ulong, 16),                        # 视频宽
        ("channels", ctypes.c_ulong, 16),                    # 视频的通道
        ("FaceData", ctypes.c_char, [4*480*320]),            # 视频数据
    ]

# open share_memory
shared_memory = ctypes.CDLL("lib.so.6")
shared_memory.open_shared_memory()

# read struct
shared_data = shared_memory.get_shared_data()
wartering_cart_data = _SmartADASExchangeInfo.from_buffer(shared_data)

print("RoadLine", wartering_cart_data.RoadLine)

# change data (lane_nums)
wartering_cart_data.RoadLine = 2

shared_memory.close_shared_memory()