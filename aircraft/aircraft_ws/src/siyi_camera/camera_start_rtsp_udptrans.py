"""
@file test_rtsp.py
@Description: Test receiving RTSP stream from IP camera
@Author: Mohamed Abdelkader
@Contact: mohamedashraf123@gmail.com
All rights reserved 2022
"""

import sys
import os
  
current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
  
sys.path.append(parent_directory)

from stream import SIYIRTSP
from siyi_sdk import SIYISDK
from stream import UDPSender
from time import sleep
import cv2

def test():

    cam = SIYISDK(server_ip="192.168.144.25", port=37260)
    if not cam.connect():
        print("No connection ")
        exit(1)

    # Get camera name
    cam_str = cam.getCameraTypeString()
    cam.disconnect()
    
    rtsp = SIYIRTSP(rtsp_url="rtsp://192.168.144.25:8554/main.264",debug=False, cam_name=cam_str)
    rtsp.setShowWindow(False)

    udp_sender = UDPSender(host='127.0.0.1', port=5600)
    udp_sender.start()

    try:
        # 主循环：从 RTSP 客户端获取帧，推送到 UDP 发送器
        while(True):
            # 获取最新的帧
            frame = rtsp.getFrame()
            
            # 推送帧到 UDP 管道
            udp_sender.setFrame(frame)
            
            # 保持主线程运行
            sleep(0.005) 
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    
    # 3. 清理
    finally:
        rtsp.close()
        udp_sender.stop()
        cv2.destroyAllWindows()
        exit(0)

if __name__ == "__main__":
    test()