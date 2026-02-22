#!/usr/bin/env python3
"""
main_copy.py ocr识别功能测试程序

组合策略
"""

import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time
from tools.visual_tracker import VisualTracker
from tools.obstacle_end import ForwardObstacleApproacher  # 导入点云停止工具
from tools.obstacle_tracker import ObstacleTracker  # 导入点云追踪工具
from tools.align_and_detect import align_and_detect  # 导入方向调整与目标检测功能
from tools.color_tracker import ColorTracker  # 导入方向调整与目标检测功能
# from tools.GrayTracker import GrayTracker # 导入终点辅助矫正

# ───────── Twist 助手函数 ───────────────────────────────────────────
def make_twist(x=0.0, y=0.0, z=0.0):
    t = Twist()
    t.linear.x = x
    t.linear.y = y
    t.angular.z = z  # 绕Z轴的偏航角(yaw)使用传入的值
    return t


# ───────── 主节点 ─────────────────────────────────────────────────
class MainDemo(Node):
    def __init__(self):
        super().__init__("main_demo")
        self.cmd_pub = self.create_publisher(Twist, "/robot0/cmd_vel", 10)
        self.direction_sub = self.create_subscription(
            String, "/robot0/direction_cmd", self.direction_callback, 10
        )
        
        # OCR相关变量
        self.ocr_result = ""
        self.ocr_received = False
        self.ocr_timeout = 5.0  # 超时时间(秒)
        
        time.sleep(1.0)  # 等待发布器连接

    def direction_callback(self, msg:String):
        """处理方向指令回调"""
        self.ocr_result = msg.data
        self.ocr_received = True
        self.get_logger().info(f"收到方向指令: {self.ocr_result}")

    def find_direction_ocr(self):
        self.get_logger().info("开始判断")
        self.ocr_received = False
        timeout_sec = 5.0
        start = time.time()
        while rclpy.ok() and not self.ocr_received:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > timeout_sec:
                self.get_logger().warn("OCR 识别超时")
                break

        if self.ocr_received:
            self.get_logger().info(f"已保存 OCR 内容: {self.ocr_result}")
        self.get_logger().info("结束判断")
        

    def execute_motion(self, duration, twist_cmd, description):
        """执行指定动作"""
        self.get_logger().info(f"执行动作: {description}")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            self.cmd_pub.publish(twist_cmd)
            time.sleep(0.1)  # 持续发送命令
        
        # 发送停止命令
        self.cmd_pub.publish(make_twist())
        self.get_logger().info(f"动作完成: {description}")

# ───────── 主函数 ────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    demo = MainDemo()
    
    tracker = ObstacleTracker(context=demo.context)

    # visual_tracker = VisualTracker()
    
    try:
        demo.execute_motion(16,make_twist(x=-0.5),"后退")  # 后退准备ocr识别
        time.sleep(1.5)
        
        demo.find_direction_ocr()                         # ocr识别
        time.sleep(2)

        demo.get_logger().info("启动点云追踪...")           # 点云追踪到避障障碍物前
        tracker.update_parameters()
        tracker.track_until_reached()
        demo.get_logger().info("点云追踪结束")
        time.sleep(1.5)

        if demo.ocr_result=="right":                      # 进行越障,默认向左(保证未识别到文字时不影响其他任务的完成)
            demo.execute_motion(5.6, make_twist(y=-0.8), "右移")
            time.sleep(2)
            demo.execute_motion(8.4, make_twist(x=0.8), "前行")
            time.sleep(2)
            demo.execute_motion(3.6, make_twist(y=0.8), "左移")
            time.sleep(2)
        else:
            demo.execute_motion(5.4, make_twist(y=0.8), "向左平移")
            # time.sleep(2)
            demo.execute_motion(7.8, make_twist(x=0.8), "向前行走")
            # time.sleep(2)
            demo.execute_motion(4.0, make_twist(y=-0.8), "向右平移")
            time.sleep(2)


        align_and_detect(context=demo.context)            # 识别任务1图片
        time.sleep(2.0)
        demo.execute_motion(8, make_twist(x=0.5,z=1.0), "逆时针转弯 180度-1") # 转弯,跳过越障任务,但识别越障图片
        align_and_detect(context=demo.context, enable_alignment=False)
        demo.execute_motion(13, make_twist(x=0.5,z=1.0), "逆时针转弯 180度-2")
        time.sleep(1.0)

        demo.execute_motion(3,make_twist(x=0.5), "前") # 并入识别任务2前进轨道
        time.sleep(1.0)

        demo.execute_motion(8,make_twist(y=-0.5), "右移") # 并入识别任务2前进轨道
        time.sleep(1.0)

        demo.get_logger().info("启动点云追踪...151515")    # 追踪前进到识别任务2图片前
        tracker.update_parameters(obstacle_threshold=1.5,sideway_speed=0.3,fov_width=0.5,max_distance=5,fixed_orientation=math.pi)
        tracker.track_until_reached()
        demo.get_logger().info("点云追踪结束151515")
        time.sleep(1.0)

        align_and_detect(context=demo.context)           # 识别任务2图片
        time.sleep(1.0)

        demo.get_logger().info("启动点云追踪...")          # 靠近识别2图片,转弯准备
        tracker.update_parameters(obstacle_threshold=0.8)
        tracker.track_until_reached()
        demo.get_logger().info("点云追踪结束")
        time.sleep(1.0)

        demo.execute_motion(8, make_twist(x=0.5,z=-1.0), "顺时针 90度") #转弯, 准备前往识别任务3
        time.sleep(1.0)

        demo.get_logger().info("启动点云追踪...")
        tracker.update_parameters(obstacle_threshold=1.0,fixed_orientation=math.pi / 2) #追踪前进到识别任务3图片前
        tracker.track_until_reached()
        demo.get_logger().info("点云追踪结束")
        time.sleep(1.0)

        align_and_detect(context=demo.context)           # 识别任务3图片
        time.sleep(1.0)

        demo.execute_motion(10, make_twist(x=0.5,z=-1.0), "顺时针 90度") #转弯, 准备前往终点位置
        time.sleep(1.0)
        demo.execute_motion(4,make_twist(x=0.6),"前行")

        demo.get_logger().info("开始视觉追踪红色物体")      # 追踪前往终点位置
        visual_tracker = ColorTracker(context=demo.context)
        visual_tracker.track_until_lost()
        demo.get_logger().info("视觉追踪结束")
        time.sleep(1.0)

        demo.execute_motion(18,make_twist(x=0.6),"前行终点")
        time.sleep(1.0)
        # demo.execute_motion(11,make_twist(z=-0.8),"turn right")

        
        # demo.get_logger().info("开始视觉追踪灰色物体")
        # gray_tracker = GrayTracker() 
        # gray_tracker.start_tracking()
        # rclpy.spin(tracker)
        # demo.get_logger().info("追踪结束")

        
    
        demo.get_logger().info("✅ 演示程序完成")
        
    finally:
        # 确保机器人停止
        demo.cmd_pub.publish(make_twist())
        tracker.destroy_node()
        demo.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()