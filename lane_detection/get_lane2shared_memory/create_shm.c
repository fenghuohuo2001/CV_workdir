#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>

#define SHMSIZE 1024
typedef struct _SmartADASExchangeInfo
{
	//STM32上报的数据
	unsigned short temperature;    //温度 20
	unsigned short humidity;    	 //湿度  80
	unsigned short BDoor:1,     	 //后炮 0
	               RBDDoor:1,         //右后下0
	               RBUDoor:1,     //右后上
	               RFBDoor:1,     //右前
	               LBDDoor:1,      //左后下
	               LBUoor:1,      //左上
	               LFDoor:1,     //右前
	               FDoor:1,     //前炮 0 1
	             //  WaterDev:2,       //水炮阀门
	               ManAdjust:2,       //手动调节标识,置为1，发送0x02数据包  0 1
	               mLeftCamera:2,     //左侧摄像头状态给
	               mRightCamera:2,    //右侧侧摄像头状态给
	               CameraSwitch:2;    //1-后置摄像头工作 2-前置摄像头的车道线识别工作 3-前置摄像头植物检测工作           2
	unsigned short WaterLevel;    //温度
	unsigned short PressLevel;    	 //压力
	unsigned long electronDev:8,     //电磁阀开度 0%~100%对应0x00~0x64
	              MainflowVal:8,    //换算成对应的十进制/100=实际值单位L/s
	              SecflowVal:8,     //回流管道流量2：换算成对应的十进制/100=实际值单位L/s
	              pitchAng:8;       //水炮俯仰角度：0~180度
	unsigned long horizAng:8,       //水炮左右转动角度：0~180度
	              LeftPerson:2,     //左侧侧有行人 0  1
	              RightPerson:2,  //右侧有行人
	              RoadLine:4,     //当前车道数：4车道0x04，8车道0x08(后续可能要改成有符号的类型)
	              horizDis:8,     //植物与摄像头之间的水平距离
	              verticalDis:8;  //植物与摄像头之间的垂直距离
	unsigned long DevLen:4,       //炮口伸长0/4对应0x00、1/4对应0x01、2/4对应0x02、3/4对应0x03、4/4对应0x04
	              LastLeftPerson:2,   //上一次左侧侧有行人 0
	              LastRightPerson:2,  //上一次右侧有行人
	              LastRoadLine:4,     //上一次当前车道数：4车道0x04，8车道0x08
	              LasthorizDis:8,     //上一次植物与摄像头之间的水平距离
	              LastverticalDis:8,  //上一次植物与摄像头之间的垂直距离
	              LastDevLen:4;        //上一次炮口伸长0/4对应0x00、1/4对应0x01、2/4对应0x02、3/4对应0x03、4/4对应0x04
	unsigned long RPersonWarn:2,        	//右侧行人报警
	              LPersonWarn:2;           //左侧行人报警

	unsigned long showDsm:8,       //显示模式 0xA1-DSM 0xA2-后置 0XA3 左侧 0xA4 右侧
	              PicDataValid:8,  //视频帧是否正常
		      rows:16;         //视频高
	unsigned long cols:16,          //视频宽
		      channels:16;      //视频的通道
	unsigned char FaceData[4*480*320];   //视频数据

}SmartADASExchangeInfo;

int main(){

	int shm_id;

	shm_id = shmget((key_t)2000,SHMSIZE,IPC_CREAT|0666);
	if(shm_id<0){
	perror("shmget error!");
	return -1;
	}
	printf("shm_id :%d\n",shm_id);

}
