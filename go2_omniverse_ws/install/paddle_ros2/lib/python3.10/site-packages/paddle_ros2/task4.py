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
    (0.9, 0.5),  # 第二个目标点
    (2.0, 0.5),
    (1.9, 0.0),
    (3.0, 0.0),
    (2.9, 0.5),
]

# 到达目标点的容差
TOLERANCE = 0.1

# 横向移动的允许偏差
SIDESTEP_TOLERANCE = 0.1

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

# =====================================================================
# 导航函数
# =====================================================================
def navigate_to_goal(cmd_pub, odom_sub, goal_position, tolerance=TOLERANCE, sidestep_tolerance=SIDESTEP_TOLERANCE):
    """
    导航到指定点位
    
    参数:
        cmd_pub: ROS 发布器
        odom_sub: ROS 订阅器 (Odometry)
        goal_position: 目标点位 (x, y)
        tolerance: 到达目标的容差
        sidestep_tolerance: 横向移动的允许偏差
    """
    def odom_callback(msg):
        nonlocal current_position, current_yaw
        current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        # 计算当前机器人的朝向（yaw）
        current_yaw = euler_from_quaternion(msg.pose.pose.orientation)

    def euler_from_quaternion(quaternion):
        """
        从四元数中提取欧拉角（仅返回yaw角）
        """
        import math
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return yaw

    current_position = (0.0, 0.0)
    current_yaw = 0.0
    odom_sub.callback = odom_callback
    
    goal_x, goal_y = goal_position
    while True:
        current_x, current_y = current_position
        distance_to_goal = math.sqrt((goal_x - current_x) ** 2 + (goal_y - current_y) ** 2)
        
        if distance_to_goal < tolerance:
            print(f"到达目标点位: {goal_position}")
            break
        
        # 计算目标方向
        angle_to_goal = math.atan2(goal_y - current_y, goal_x - current_x)
        
        # 计算需要调整的角度
        angle_diff = angle_to_goal - current_yaw
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # 归一化到[-pi, pi]
        
        # 检查是否可以直接横向移动
        if abs(angle_diff) < sidestep_tolerance or abs(angle_diff) > (math.pi - sidestep_tolerance):
            # 目标点在正左右侧，直接横向移动
            if goal_y > current_y:
                cmd_pub.publish(create_twist(y=DEFAULT_VELOCITY))  # 向左横向移动
            else:
                cmd_pub.publish(create_twist(y=-DEFAULT_VELOCITY))  # 向右横向移动
        else:
            # 调整方向
            if abs(angle_diff) > 0.05:  # 如果角度差大于0.05弧度，则调整方向
                if angle_diff > 0:
                    cmd_pub.publish(create_twist(z=0.2))  # 左转
                else:
                    cmd_pub.publish(create_twist(z=-0.2))  # 右转
            else:
                cmd_pub.publish(create_twist(x=DEFAULT_VELOCITY))  # 向前移动
        
        time.sleep(0.1)  # 等待0.1秒后再次检查

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