#!/usr/bin/env python3
"""
task.py · Modular task control with sequential state machine
添加了原地旋转到地图X轴方向的功能测试
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
ROTATION_SPEED = 0.3    # 旋转速度

# 目标点位列表 (x, y, z)
WAYPOINTS = [
    (0.2, 0.0, 0.0),  # 第一个目标点
    (0.9, 0.5, 0.5),  # 第二个目标点
    (2.0, 0.5, 0.0),
    (1.9, 0.0, -0.5),
    (3.0, 0.0, 0.0),
    (2.9, 0.5, 0.5),
]

# 到达目标点的容差
TOLERANCE = 0.15

# 调整角度的容差
ANGLE_TOLERANCE = 0.05

# 超时时间（秒）
TIMEOUT = 50.0

# 创建 Twist 消息
def create_twist(x=0.0, y=0.0, z=0.0):
    t = Twist()
    t.linear.x = x
    t.linear.y = y
    t.angular.z = z
    return t

# 从四元数中提取欧拉角（仅返回yaw角）
def euler_from_quaternion(quaternion):
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return yaw

# 调整机器人朝向到目标方向
def adjust_orientation(cmd_pub, current_yaw, goal_yaw, tolerance=ANGLE_TOLERANCE, timeout=TIMEOUT):
    start_time = time.time()
    while True:
        angle_diff = goal_yaw - current_yaw
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # 归一化到[-pi, pi]
        
        if abs(angle_diff) < tolerance:
            cmd_pub.publish(create_twist())  # 停止
            print(f"朝向调整完成，目标朝向: {goal_yaw:.2f} 弧度")
            break
        
        if time.time() - start_time > timeout:
            cmd_pub.publish(create_twist())  # 停止
            print("朝向调整超时")
            raise TimeoutError("调整方向超时")
        
        if angle_diff > 0:
            cmd_pub.publish(create_twist(z=0.2))  # 左转
            print("正在左转...")
        else:
            cmd_pub.publish(create_twist(z=-0.2))  # 右转
            print("正在右转...")
        
        time.sleep(0.1)

# 移动机器人到目标点
def move_to_goal(cmd_pub, current_position, goal_position, tolerance=TOLERANCE, timeout=TIMEOUT):
    start_time = time.time()
    while True:
        current_x, current_y = current_position
        goal_x, goal_y = goal_position
        distance_to_goal = math.sqrt((goal_x - current_x) ** 2 + (goal_y - current_y) ** 2)
        
        if distance_to_goal < tolerance:
            cmd_pub.publish(create_twist())  # 停止
            print(f"到达目标点: {goal_position}")
            break
        
        if time.time() - start_time > timeout:
            cmd_pub.publish(create_twist())  # 停止
            print("移动到目标点超时")
            raise TimeoutError("移动到目标点超时")
        
        if goal_x > current_x:
            cmd_pub.publish(create_twist(x=DEFAULT_VELOCITY))  # 向前移动
            print("正在向前移动...")
        elif goal_x < current_x:
            cmd_pub.publish(create_twist(x=-DEFAULT_VELOCITY))  # 向后移动
            print("正在向后移动...")
        elif goal_y > current_y:
            cmd_pub.publish(create_twist(y=DEFAULT_VELOCITY))  # 向左移动
            print("正在向左移动...")
        elif goal_y < current_y:
            cmd_pub.publish(create_twist(y=-DEFAULT_VELOCITY))  # 向右移动
            print("正在向右移动...")
        
        time.sleep(0.1)

# 导航到指定点位
def navigate_to_goal(cmd_pub, odom_sub, goal_position, goal_yaw, tolerance=TOLERANCE, timeout=TIMEOUT):
    def odom_callback(msg):
        nonlocal current_position, current_yaw
        current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        current_yaw = euler_from_quaternion(msg.pose.pose.orientation)
    
    current_position = (0.0, 0.0)
    current_yaw = 0.0
    odom_sub.callback = odom_callback
    
    print(f"开始导航到目标点: {goal_position}, 目标朝向: {goal_yaw:.2f} 弧度")
    # 调整朝向到目标方向
    adjust_orientation(cmd_pub, current_yaw, goal_yaw, timeout=timeout)
    
    # 移动到目标点
    move_to_goal(cmd_pub, current_position, goal_position, timeout=timeout)
    
    # 再次调整朝向到目标方向
    adjust_orientation(cmd_pub, current_yaw, goal_yaw, timeout=timeout)

# 任务控制节点
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
        self.current_yaw = 0.0
    
    def odom_callback(self, msg):
        """存储当前位置和朝向"""
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.current_yaw = euler_from_quaternion(msg.pose.pose.orientation)
    
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
    
    def rotate_to_x_axis(self):
        """旋转到地图坐标系X轴方向（0弧度）"""
        if self.running:
            self.get_logger().warn("任务已在运行中")
            return
        
        self.running = True
        self.task_thread = threading.Thread(target=self._rotate_to_x_axis_impl)
        self.task_thread.daemon = True
        self.task_thread.start()
        self.get_logger().info("开始旋转到X轴方向!")
    
    def _rotate_to_x_axis_impl(self):
        """实现旋转到X轴方向的逻辑"""
        try:
            # 目标方向为0弧度（地图坐标系X轴方向）
            target_yaw = 0.0
            
            # 显示当前朝向
            current_yaw_deg = math.degrees(self.current_yaw)
            self.get_logger().info(f"当前朝向: {current_yaw_deg:.2f}度")
            
            # 计算旋转角度差（归一化到[-π, π]）
            angle_diff = target_yaw - self.current_yaw
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
            
            # 显示旋转信息
            self.get_logger().info(f"需要旋转: {math.degrees(angle_diff):.2f}度")
            
            # 执行旋转
            start_time = time.time()
            while abs(angle_diff) > ANGLE_TOLERANCE and (time.time() - start_time) < TIMEOUT:
                # 计算当前角度差
                angle_diff = target_yaw - self.current_yaw
                angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
                
                # 发布旋转命令
                if angle_diff > 0:
                    self.cmd_pub.publish(create_twist(z=ROTATION_SPEED))  # 逆时针旋转
                else:
                    self.cmd_pub.publish(create_twist(z=-ROTATION_SPEED))  # 顺时针旋转
                
                time.sleep(0.1)
            
            # 停止旋转
            self.cmd_pub.publish(create_twist())
            
            # 显示结果
            if abs(angle_diff) <= ANGLE_TOLERANCE:
                self.get_logger().info("✅ 旋转完成! 当前朝向X轴正方向")
            else:
                self.get_logger().warn("⚠️ 旋转超时")
            
        except Exception as e:
            self.get_logger().error(f"旋转出错: {e}")
        finally:
            self.running = False
    
    def task_sequence(self):
        """定义任务执行序列 - 可轻松扩展"""
        logger = self.get_logger()
        
        try:
            for waypoint in WAYPOINTS:
                goal_position = (waypoint[0], waypoint[1])
                goal_yaw = waypoint[2]
                logger.info(f"导航到目标点位: {goal_position}, 目标朝向: {goal_yaw:.2f} 弧度")
                navigate_to_goal(self.cmd_pub, self.odom_sub, goal_position, goal_yaw)
            
            # 任务完成
            logger.info("✅ 任务完成!")
            
        except Exception as e:
            logger.error(f"任务执行出错: {e}")
        finally:
            self.running = False

# 主执行入口
def main(args=None):
    rclpy.init(args=args)
    controller = TaskController()
    
    # 简单命令行界面
    def user_input():
        while rclpy.ok():
            print("\n===== 命令菜单 =====")
            print("1: 运行完整任务序列")
            print("2: 旋转到地图X轴方向")
            print("3: 退出程序")
            cmd = input("请输入命令: ")
            
            if cmd == "1":
                controller.start_task()
            elif cmd == "2":
                controller.rotate_to_x_axis()
            elif cmd == "3":
                break
            else:
                print("无效命令，请重新输入")
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