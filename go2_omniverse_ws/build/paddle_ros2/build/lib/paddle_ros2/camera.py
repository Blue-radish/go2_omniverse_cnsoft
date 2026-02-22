#!/usr/bin/env python3
"""
capture_on_enter.py - 按回车键拍照工具
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import os
import time
import threading
import numpy as np

class InteractiveImageCapture(Node):
    def __init__(self):
        super().__init__("interactive_image_capture")
        
        # 订阅图像
        self.image_sub = self.create_subscription(
            Image, '/robot0/front_cam/rgb', self.image_callback, 10)
        
        # 参数设置
        self.result_dir = "./result"  # 结果保存目录
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 当前图像缓存
        self.latest_image = None
        self.image_lock = threading.Lock()
        
        # 启动用户输入线程
        self.input_thread = threading.Thread(target=self.user_input_handler, daemon=True)
        self.input_thread.start()
        
        self.get_logger().info("交互式图像捕获已初始化")
        self.get_logger().info("按回车键拍照，输入 'q' 退出程序")
    
    def user_input_handler(self):
        """处理用户输入"""
        while rclpy.ok():
            user_input = input()
            
            if user_input.lower() == 'q':
                self.get_logger().info("退出程序...")
                rclpy.shutdown()
                return
            
            # 按回车键拍照
            self.capture_image()
    
    def capture_image(self):
        """捕获并保存当前图像"""
        with self.image_lock:
            if self.latest_image is None:
                self.get_logger().info("尚未收到图像数据，请稍后...")
                return
            
            try:
                # 转换图像数据
                dtype = np.uint8
                image_np = np.frombuffer(self.latest_image.data, dtype=dtype)
                
                # 处理图像格式
                if self.latest_image.encoding == 'rgb8' or self.latest_image.encoding == 'bgr8':
                    image_np = image_np.reshape((self.latest_image.height, self.latest_image.width, 3))
                    if self.latest_image.encoding == 'rgb8':
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                else:
                    self.get_logger().warn(f"不支持的图像格式: {self.latest_image.encoding}")
                    return
                
                # 保存图像
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(self.result_dir, f"capture_{timestamp}.jpg")
                cv2.imwrite(img_path, image_np)
                self.get_logger().info(f"📸 图像已保存至: {img_path}")
                
            except Exception as e:
                self.get_logger().error(f"图像处理失败: {e}")
    
    def image_callback(self, msg):
        """更新最新图像"""
        with self.image_lock:
            self.latest_image = msg

def main():
    rclpy.init()
    node = InteractiveImageCapture()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()