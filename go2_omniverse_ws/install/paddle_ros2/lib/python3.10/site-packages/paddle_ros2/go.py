#!/usr/bin/env python3
"""
go2_obstacle_tracking.py  ·  Unitree Go2障碍物追踪（改进版）

优化点：
1. 引入状态机机制（旋转->前进->停止）
2. 添加动作完成标志，避免频繁发布命令
3. 增加旋转和前进的持续执行逻辑
4. 优化控制循环结构
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np
import time

# ───────── 状态枚举 ──────────────────────────────────────────
class State:
    SEARCHING = 0      # 寻找障碍物
    ALIGNING = 1       # 旋转调整方向
    APPROACHING = 2    # 向障碍物前进
    REACHED = 3        # 到达目标距离

# ───────── 创建Twist消息的辅助函数 ─────────────────────────────
def make_twist(x=0.0, y=0.0, z=0.0):
    t = Twist()
    t.linear.x = x
    t.linear.y = y
    t.angular.z = z
    return t

# ───────── ROS 2障碍物追踪节点（改进版）───────────────────────
class Go2ObstacleTracking(Node):
    def __init__(self):
        super().__init__("go2_obstacle_tracking")
        self.pub = self.create_publisher(Twist, "/robot0/cmd_vel", 10)
        
        # 订阅点云和里程计话题
        self.point_cloud_sub = self.create_subscription(
            PointCloud2, "/robot0/point_cloud2", self.point_cloud_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/robot0/odom", self.odom_callback, 10)
        
        # 机器人当前状态
        self.robot_position = Point()
        self.robot_yaw = 0.0
        
        # 参数设置
        self.obstacle_threshold = 0.45   # 障碍物距离阈值（米）
        self.forward_speed = 0.5        # 前进速度（米/秒）
        self.turn_speed = 1           # 转向速度（弧度/秒）
        self.fov_width = 0.5            # 视野宽度（米）
        self.min_points = 5              # 有效障碍物所需的最小点数
        self.center_deadzone = 0.05      # 中心点死区阈值（米）
        self.ground_threshold = 0.1      # 地面高度阈值（米）
        
        # 障碍物追踪状态
        self.state = State.SEARCHING
        self.obstacle_detected = False
        self.obstacle_center = None
        self.obstacle_distance = float('inf')
        self.last_cmd_time = self.get_clock().now()
        self.current_cmd = make_twist()
        
        self.get_logger().info("启动Go2障碍物追踪（改进版）...")
        self.get_logger().info(f"追踪视野宽度: {self.fov_width}米")
        self.get_logger().info(f"地面高度阈值: {self.ground_threshold}米")
        
        # 启动控制循环
        self.timer = self.create_timer(0.1, self.control_loop)  # 10Hz控制循环

    def odom_callback(self, msg):
        """更新机器人位置和方向"""
        self.robot_position = msg.pose.pose.position
        
        # 从四元数计算偏航角（yaw）
        orientation = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y**2 + orientation.z**2)
        self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)
        
    def point_cloud_callback(self, msg):
        """处理点云数据，检测前方障碍物并计算中心点"""
        try:
            # 提取点云数据
            points = point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)
            
            # 收集视野内的点（前方且横向在视野宽度内）
            fov_points = []
            min_distance = float('inf')
            
            for p in points:
                # 将点转换到机器人坐标系
                dx = p[0] - self.robot_position.x
                dy = p[1] - self.robot_position.y
                
                robot_frame_x = dx * np.cos(self.robot_yaw) + dy * np.sin(self.robot_yaw)
                robot_frame_y = -dx * np.sin(self.robot_yaw) + dy * np.cos(self.robot_yaw)
                
                # 只考虑机器人前方、视野宽度内且高于地面的点
                if (robot_frame_x > 0 and 
                    abs(robot_frame_y) <= self.fov_width/2 and
                    p[2] > self.ground_threshold):
                    fov_points.append((robot_frame_x, robot_frame_y))
                    
                    # 更新最小距离
                    distance = np.sqrt(robot_frame_x**2 + robot_frame_y**2)
                    if distance < min_distance:
                        min_distance = distance
            
            # 如果有足够的点，计算中心点
            if len(fov_points) >= self.min_points:
                self.obstacle_detected = True
                self.obstacle_distance = min_distance
                
                # 计算平均位置作为中心点
                avg_x = sum(p[0] for p in fov_points) / len(fov_points)
                avg_y = sum(p[1] for p in fov_points) / len(fov_points)
                self.obstacle_center = (avg_x, avg_y)
                
                self.get_logger().debug(
                    f"检测到障碍物: 中心点=({avg_x:.2f}, {avg_y:.2f}), 距离={min_distance:.2f}米", 
                    throttle_duration_sec=1.0)
            else:
                self.obstacle_detected = False
                self.obstacle_center = None
                
        except Exception as e:
            self.get_logger().error(f"处理点云数据时出错: {str(e)}")
            self.obstacle_detected = False
            self.obstacle_center = None

    def control_loop(self):
        """基于状态机的控制循环"""
        current_time = self.get_clock().now()
        
        # 状态机处理
        if self.state == State.SEARCHING:
            if self.obstacle_detected:
                if self.obstacle_distance <= self.obstacle_threshold:
                    self.state = State.REACHED
                    self.get_logger().info("已到达障碍物阈值距离，停止。")
                else:
                    self.state = State.ALIGNING
                    self.get_logger().info("检测到障碍物，开始旋转调整方向。")
            else:
                # 没有检测到障碍物，停止
                self.publish_once(make_twist())
                self.get_logger().info("视野内未检测到障碍物，停止。", throttle_duration_sec=1.0)
        
        elif self.state == State.ALIGNING:
            if not self.obstacle_detected:
                self.state = State.SEARCHING
                self.get_logger().info("调整过程中障碍物消失，返回搜索状态。")
            elif self.obstacle_distance <= self.obstacle_threshold:
                self.state = State.REACHED
                self.get_logger().info("调整过程中达到阈值距离，停止。")
            else:
                # 计算障碍物在视野中的横向偏移
                center_y = self.obstacle_center[1]
                
                # 检查障碍物是否已居中
                if abs(center_y) < self.center_deadzone:
                    self.state = State.APPROACHING
                    self.get_logger().info("障碍物已居中，开始前进。")
                else:
                    # 障碍物未居中，原地旋转调整方向
                    turn_speed = np.sign(center_y) * self.turn_speed
                    self.publish_continuous(make_twist(z=turn_speed))
                    
                    # 记录旋转方向
                    direction = "向左" if turn_speed < 0 else "向右"
                    self.get_logger().info(
                        f"障碍物偏移: {center_y:.2f}米, 原地{direction}旋转中", 
                        throttle_duration_sec=0.5)
        
        elif self.state == State.APPROACHING:
            if not self.obstacle_detected:
                self.state = State.SEARCHING
                self.get_logger().info("前进过程中障碍物消失，返回搜索状态。")
            elif self.obstacle_distance <= self.obstacle_threshold:
                self.state = State.REACHED
                self.get_logger().info("已到达障碍物阈值距离，停止。")
            else:
                # 检查障碍物是否仍然居中
                center_y = self.obstacle_center[1]
                if abs(center_y) > self.center_deadzone:
                    self.state = State.ALIGNING
                    self.get_logger().info("前进过程中障碍物偏移，重新调整方向。")
                else:
                    # 持续前进
                    self.publish_continuous(make_twist(x=self.forward_speed))
                    self.get_logger().info("向障碍物前进中...", throttle_duration_sec=0.5)
        
        elif self.state == State.REACHED:
            # 停止状态
            self.publish_once(make_twist())
            # 如果障碍物离开或距离变远，重新开始追踪
            if not self.obstacle_detected or self.obstacle_distance > self.obstacle_threshold * 1.2:
                self.state = State.SEARCHING
                self.get_logger().info("障碍物移动，重新开始追踪。")

    def publish_continuous(self, twist_msg):
        """持续发布命令直到状态改变"""
        # 每秒发布一次即可，避免频繁发送
        if (self.get_clock().now() - self.last_cmd_time).nanoseconds > 1e8:  # 100ms
            self.pub.publish(twist_msg)
            self.last_cmd_time = self.get_clock().now()
            self.current_cmd = twist_msg
    
    def publish_once(self, twist_msg):
        """仅当命令改变时发布"""
        if (twist_msg.linear.x != self.current_cmd.linear.x or
            twist_msg.angular.z != self.current_cmd.angular.z):
            self.pub.publish(twist_msg)
            self.current_cmd = twist_msg
            self.last_cmd_time = self.get_clock().now()

# ───────── 程序入口 ──────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    controller = Go2ObstacleTracking()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("用户中断节点")
    finally:
        # 确保机器人停止
        controller.pub.publish(make_twist())
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()