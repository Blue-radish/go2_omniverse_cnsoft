#!/usr/bin/env python3
"""
task.py · Modular task control with sequential state machine
运动函数基于输入方向和持续时间执行
状态序列易于扩展新步骤
"""

import time
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

# =====================================================================
# 常量配置
# =====================================================================
DEFAULT_VELOCITY = 0.5  # 默认运动速度

# 目标点位列表 (x, y)
WAYPOINTS = [
    (1.0, 0.0),  # 第一个目标点
    (1.0, 1.0),  # 第二个目标点
    (8.0, 8.0)   # 第三个目标点
]

# 到达目标点的容差
TOLERANCE = 0.1

# =====================================================================
# 运动控制函数
# =====================================================================
def create_twist(x=0.0, y=0.0, z=0.0):
    """创建 Twist 消息"""
    t = Twist()
    t.linear.x = x
    t.linear.y = y
    t.angular.z = z
    return t

def execute_movement(cmd_pub, direction, duration, velocity=DEFAULT_VELOCITY):
    """
    执行指定方向的运动
    
    参数:
        cmd_pub: ROS 发布器
        direction: 运动方向 ('forward', 'back', 'left', 'right')
        duration: 运动持续时间 (秒)
        velocity: 运动速度 (可选)
    """
    start_time = time.time()
    
    # 映射方向到运动命令
    if direction == "forward":
        cmd = create_twist(x=velocity)
    elif direction == "back":
        cmd = create_twist(x=-velocity)
    elif direction == "left":
        cmd = create_twist(z=velocity)  # 正值为左转
    elif direction == "right":
        cmd = create_twist(z=-velocity)  # 负值为右转
    else:  # 未知方向默认为前进
        cmd = create_twist(x=velocity)
    
    # 执行运动
    while (time.time() - start_time) < duration:
        cmd_pub.publish(cmd)
        time.sleep(0.05)  # 20Hz控制频率
    
    cmd_pub.publish(create_twist())  # 停止运动
    time.sleep(1.0)  # 确保停止

# =====================================================================
# 导航函数
# =====================================================================
def navigate_to_goal(cmd_pub, odom_sub, goal_position, tolerance=TOLERANCE):
    """
    导航到指定点位
    
    参数:
        cmd_pub: ROS 发布器
        odom_sub: ROS 订阅器 (Odometry)
        goal_position: 目标点位 (x, y)
        tolerance: 到达目标的容差
    """
    def odom_callback(msg):
        nonlocal current_position
        current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
    
    current_position = (0.0, 0.0)
    odom_sub.callback = odom_callback
    
    goal_x, goal_y = goal_position
    while True:
        current_x, current_y = current_position
        distance_to_goal = math.sqrt((goal_x - current_x) ** 2 + (goal_y - current_y) ** 2)
        
        if distance_to_goal < tolerance:
            print(f"到达目标点位: {goal_position}")
            break
        
        # 计算方向
        angle_to_goal = math.atan2(goal_y - current_y, goal_x - current_x)
        angle_to_goal = math.degrees(angle_to_goal)
        
        # 调整方向
        if angle_to_goal > 0:
            execute_movement(cmd_pub, "left", 3.0, velocity=0.5)
        else:
            execute_movement(cmd_pub, "right", 3.0, velocity=0.5)
        
        # 向前移动
        execute_movement(cmd_pub, "forward",0.5)

# =====================================================================
# 任务控制节点
# =====================================================================
class TaskController(Node):
    def __init__(self):
        super().__init__("task_controller")
        
        # ROS 通信
        self.cmd_pub = self.create_publisher(Twist, "/robot0/cmd_vel", 10)
        self.odom_sub = self.create_subscription(Odometry, "/robot0/odom", self.odom_callback, 10)
        
        # 任务状态
        self.task_thread = None
        self.running = False
        self.current_position = (0.0, 0.0)
    
    def odom_callback(self, msg):
        """存储当前位置"""
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
    
    def start_task(self):
        """启动任务序列"""
        if self.running:
            self.get_logger().warn("任务已在运行中")
            return
        
        self.running = True
        self.task_thread = threading.Thread(target=self.task_sequence)
        self.task_thread.daemon = True
        self.task_thread.start()
        self.get_logger().info("任务启动!")
    
    def task_sequence(self):
        """定义任务执行序列 - 可轻松扩展"""
        logger = self.get_logger()
        
        try:
            for waypoint in WAYPOINTS:
                logger.info(f"导航到目标点位: {waypoint}")
                navigate_to_goal(self.cmd_pub, self.odom_sub, waypoint)
            
            # 任务完成
            logger.info("✅ 任务完成!")
            
        except Exception as e:
            logger.error(f"任务执行出错: {e}")
        finally:
            self.running = False

# =====================================================================
# 主执行入口
# =====================================================================
def main(args=None):
    rclpy.init(args=args)
    controller = TaskController()
    
    # 简单命令行界面 (在实际应用中可替换为ROS服务)
    def user_input():
        while rclpy.ok():
            cmd = input("输入命令 (输入 '1' 运行任务): ")
            if cmd == "1":
                controller.start_task()
            elif cmd == "2":
                break
            time.sleep(0.1)
    
    # 启动命令行界面线程
    input_thread = threading.Thread(target=user_input)
    input_thread.daemon = True
    input_thread.start()
    
    # 运行ROS节点
    rclpy.spin(controller)
    
    # 清理
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()