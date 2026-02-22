#!/usr/bin/env python3
"""
point_tracker.py - 手动目标点追踪工具类（支持固定朝向版）

功能:
1. 订阅里程计数据
2. 保持固定朝向（如X轴正方向）
3. 根据手动设置的目标点计算中心点
4. 如果目标点不居中，则左右平移使其居中（同时保持固定朝向）
5. 在目标点中心点居中后向目标点前进直到连续多次检测到距离达到阈值
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import numpy as np
import math

# ───────── 状态枚举 ──────────────────────────────────────────
class State:
    SEARCHING = 0      # 寻找目标点
    ALIGNING = 1       # 旋转调整方向
    APPROACHING = 2    # 向目标点前进
    REACHED = 3        # 到达目标距离
    MOVING_SIDEWAYS = 4  # 横向移动使目标居中
    ORIENTING = 5       # 调整机器人朝向

# ───────── 创建Twist消息的辅助函数 ─────────────────────────────
def make_twist(x=0.0, y=0.0, z=0.0):
    t = Twist()
    t.linear.x = x
    t.linear.y = y
    t.angular.z = z
    return t

# ───────── 目标点追踪工具类 ─────────────────────────────────────
class PointTracker(Node):
    def __init__(self, 
                 context=None,
                 cmd_vel_topic="/robot0/cmd_vel",
                 odom_topic="/robot0/odom"):
        """
        初始化目标点追踪器
        """
        super().__init__("point_tracker", context=context)
        
        # 创建发布者
        self.pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        
        # 订阅里程计话题
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10)
        
        # 机器人当前状态
        self.robot_position = Point()
        self.robot_yaw = 0.0
        
        # 目标点追踪状态
        self.state = State.SEARCHING
        self.target_point = None  # 手动设置的目标点
        self.target_distance = float('inf')
        self.target_offset = 0.0
        self.last_cmd_time = self.get_clock().now()
        self.current_cmd = make_twist()
        self.reached_count = 0          # 连续达到阈值的计数器
        self.active = False              # 追踪是否激活
        
        # 新增状态变量
        self.orientation_correction_needed = False  # 是否需要纠正朝向
        
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
        
    def _normalize_angle(self, angle):
        """将角度归一化到[-π, π]区间"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
        
    def calculate_target_relative(self):
        """计算目标点在机器人坐标系中的相对位置"""
        if self.target_point is None:
            return None, None
            
        # 计算目标点在机器人坐标系中的位置
        dx = self.target_point.x - self.robot_position.x
        dy = self.target_point.y - self.robot_position.y
        
        robot_frame_x = dx * np.cos(self.robot_yaw) + dy * np.sin(self.robot_yaw)
        robot_frame_y = -dx * np.sin(self.robot_yaw) + dy * np.cos(self.robot_yaw)
        
        distance = np.sqrt(robot_frame_x**2 + robot_frame_y**2)
        offset = robot_frame_y
        
        return distance, offset

    def control_loop(self):
        """基于状态机的控制循环"""
        if not self.active or self.target_point is None:
            return
            
        # 计算目标点相对位置
        self.target_distance, self.target_offset = self.calculate_target_relative()
        
        # 检查是否需要纠正朝向（优先级最高）
        if self.fixed_orientation is not None and self.orientation_correction_needed:
            angle_diff = self._normalize_angle(self.fixed_orientation - self.robot_yaw)
            if abs(angle_diff) > self.orientation_tolerance:
                # 进入朝向纠正状态
                self.state = State.ORIENTING
                
                # 计算旋转方向
                turn_direction = 1.0 if angle_diff > 0 else -1.0
                self.publish_continuous(make_twist(z=turn_direction * self.turn_speed))
                return
            else:
                # 朝向已正确，重置标志
                self.orientation_correction_needed = False
                
        # 状态机处理
        if self.state == State.SEARCHING:
            if self.target_distance <= self.target_threshold:
                # 连续达到阈值计数
                self.reached_count += 1
                if self.reached_count >= self.reached_count_threshold:
                    self.state = State.REACHED
            else:
                # 如果设置了固定朝向，优先使用横向移动
                if self.fixed_orientation is not None and self.sideway_speed > 0:
                    self.state = State.MOVING_SIDEWAYS
                else:
                    self.state = State.ALIGNING
        
        elif self.state == State.ALIGNING:
            if self.target_distance <= self.target_threshold:
                # 连续达到阈值计数
                self.reached_count += 1
                if self.reached_count >= self.reached_count_threshold:
                    self.state = State.REACHED
            else:
                # 检查目标点是否已居中
                if abs(self.target_offset) < self.center_deadzone:
                    self.state = State.APPROACHING
                    self.reached_count = 0  # 重置计数
                else:
                    # 目标点未居中，原地旋转调整方向
                    turn_speed = np.sign(self.target_offset) * self.turn_speed
                    self.publish_continuous(make_twist(z=turn_speed))
        
        elif self.state == State.APPROACHING:
            if self.target_distance <= self.target_threshold:
                # 连续达到阈值计数
                self.reached_count += 1
                if self.reached_count >= self.reached_count_threshold:
                    self.state = State.REACHED
            else:
                # 检查目标点是否仍然居中
                if abs(self.target_offset) > self.center_deadzone:
                    # 如果设置了固定朝向，优先使用横向移动
                    if self.fixed_orientation is not None and self.sideway_speed > 0:
                        self.state = State.MOVING_SIDEWAYS
                    else:
                        self.state = State.ALIGNING
                    self.reached_count = 0
                else:
                    # 持续前进
                    self.publish_continuous(make_twist(x=self.forward_speed))
        
        elif self.state == State.REACHED:
            # 停止状态
            self.publish_once(make_twist())
            # 追踪完成，重置状态
            self.active = False
            self.state = State.SEARCHING
            
        elif self.state == State.MOVING_SIDEWAYS:
            if abs(self.target_offset) < self.center_deadzone:
                self.state = State.APPROACHING
            else:
                # 计算横向移动方向（保持固定朝向）
                move_direction = 1 if self.target_offset > 0 else -1
                self.publish_continuous(make_twist(y=move_direction * self.sideway_speed))
        
        elif self.state == State.ORIENTING:
            # 检查是否已经完成朝向纠正
            angle_diff = self._normalize_angle(self.fixed_orientation - self.robot_yaw)
            if abs(angle_diff) <= self.orientation_tolerance:
                # 纠正完成，返回之前的状态
                self.orientation_correction_needed = False
                self.state = State.SEARCHING
            else:
                # 继续纠正朝向
                turn_direction = 1.0 if angle_diff > 0 else -1.0
                self.publish_continuous(make_twist(z=turn_direction * self.turn_speed))

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
            twist_msg.angular.z != self.current_cmd.angular.z or
            twist_msg.linear.y != self.current_cmd.linear.y):
            self.pub.publish(twist_msg)
            self.current_cmd = twist_msg
            self.last_cmd_time = self.get_clock().now()

    def stop_robot(self):
        """安全停止机器人"""
        cmd = Twist()
        self.pub.publish(cmd)
        self.get_logger().info("机器人已停止")
        
    def track_to_point(self, target_point, 
                      target_threshold=0.5,
                      forward_speed=0.5,
                      turn_speed=0.6,
                      sideway_speed=0.3,
                      center_deadzone=0.12,
                      reached_count_threshold=3,
                      fixed_orientation=None,
                      orientation_tolerance=0.03):
        """
        一行调用：追踪到指定点（阻塞式）
        
        参数:
            target_point: 地图坐标系中的目标点 (geometry_msgs/Point)
            target_threshold: 到达目标的距离阈值 (米)
            forward_speed: 前进速度 (米/秒)
            turn_speed: 旋转速度 (弧度/秒)
            sideway_speed: 横向移动速度 (米/秒)
            center_deadzone: 中心点死区宽度 (米)
            reached_count_threshold: 连续达到阈值的计数阈值
            fixed_orientation: 固定朝向 (弧度, None表示不固定)
            orientation_tolerance: 朝向容差 (弧度)
        """
        # 设置目标点
        self.target_point = target_point
        
        # 配置参数
        self.target_threshold = target_threshold
        self.forward_speed = forward_speed
        self.turn_speed = turn_speed
        self.sideway_speed = sideway_speed
        self.center_deadzone = center_deadzone
        self.reached_count_threshold = reached_count_threshold
        self.fixed_orientation = fixed_orientation
        self.orientation_tolerance = orientation_tolerance
        
        # 重置状态
        self.active = True
        self.state = State.SEARCHING
        self.reached_count = 0
        self.orientation_correction_needed = False
        
        # 如果设置了固定朝向，确保初始方向正确
        if self.fixed_orientation is not None:
            angle_diff = self._normalize_angle(self.fixed_orientation - self.robot_yaw)
            if abs(angle_diff) > self.orientation_tolerance:
                self.state = State.ORIENTING
        
        # 阻塞式追踪
        while rclpy.ok() and self.active:
            rclpy.spin_once(self, timeout_sec=0.1)
            
        # 返回是否因为达到目标而停止
        return self.state == State.REACHED