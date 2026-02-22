#!/usr/bin/env python3
"""
waypoint_navigator.py - 增强版点对点导航工具

功能:
1. 支持自定义目标点序列
2. 可配置导航参数（速度、容差、超时等）
3. 支持直接横向移动到正左/正右目标点
4. 支持添加相对点位（基于当前位姿的偏移）
5. 实现平滑的合力运动控制
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import math
import time
import threading
import numpy as np

# 创建 Twist 消息的辅助函数
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

class WaypointNavigator(Node):
    def __init__(self, 
                 waypoints=None,
                 velocity=0.5,
                 rotation_velocity=0.4,  # 新增：旋转速度参数
                 position_tolerance=0.1,
                 angle_tolerance=0.03,
                 timeout=30.0,
                 allow_lateral_movement=True,
                 lateral_threshold=0.1,
                 navigation_mode="latest",  # 新增导航模式参数
                 context=None,
                 cmd_vel_topic="/robot0/cmd_vel",
                 odom_topic="/robot0/odom"):
        """
        初始化点对点导航器
        
        参数:
            waypoints: 目标点位列表 [(x, y, yaw), ...]
            velocity: 默认运动速度
            rotation_velocity: 旋转速度（弧度/秒）
            position_tolerance: 位置容差
            angle_tolerance: 角度容差
            timeout: 超时时间（秒）
            allow_lateral_movement: 是否允许直接横向移动到正左/正右目标点
            lateral_threshold: 横向移动的X轴阈值（米）
            navigation_mode: 导航模式 ("all" - 所有点, "latest" - 只导航最新点)
            context: ROS 2 上下文对象
            cmd_vel_topic: 速度控制话题
            odom_topic: 里程计话题
        """
        super().__init__("waypoint_navigator", context=context)
        
        # 参数验证
        self._validate_params(velocity, rotation_velocity, position_tolerance, 
                              angle_tolerance, timeout)
        
        # 存储参数
        self.waypoints = waypoints or []
        self.velocity = velocity
        self.rotation_velocity = rotation_velocity  # 新增：存储旋转速度
        self.position_tolerance = position_tolerance
        self.angle_tolerance = angle_tolerance
        self.timeout = timeout
        self.allow_lateral_movement = allow_lateral_movement
        self.lateral_threshold = lateral_threshold
        self.navigation_mode = navigation_mode.lower()  # 新增：导航模式
        
        # 创建发布者
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        
        # 订阅里程计话题
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10)
        
        # 导航状态
        self.current_position = (0.0, 0.0)
        self.current_yaw = 0.0
        self.active = False
        self.navigation_thread = None
        self.navigation_complete = False
        self.navigation_failed = False
        
        # self.get_logger().info("点对点导航器已初始化")
        # self.get_logger().info(f"目标点数: {len(self.waypoints)}")
        # self.get_logger().info(f"导航速度: {self.velocity} 米/秒")
        # self.get_logger().info(f"旋转速度: {self.rotation_velocity} 弧度/秒")
        # self.get_logger().info(f"允许横向移动: {'是' if self.allow_lateral_movement else '否'}")
        # self.get_logger().info(f"导航模式: {self.navigation_mode}")
    
    def _validate_params(self, velocity, rotation_velocity, position_tolerance, 
                         angle_tolerance, timeout):
        """验证参数是否有效"""
        if velocity <= 0:
            raise ValueError("速度必须大于0")
        if position_tolerance <= 0:
            raise ValueError("位置容差必须大于0")
        if angle_tolerance <= 0:
            raise ValueError("角度容差必须大于0")
        if timeout <= 0:
            raise ValueError("超时时间必须大于0")
        if rotation_velocity <= 0:
            raise ValueError("旋转速度必须大于0")
    
    def update_parameters(self, **kwargs):
        """运行时更新参数"""
        valid_params = [
            "waypoints", "velocity", "position_tolerance", "angle_tolerance",
            "timeout", "allow_lateral_movement","lateral_threshold",
            "navigation_mode",  # 新增：支持更新导航模式
            "rotation_velocity"  # 新增：支持更新旋转速度
        ]
        
        for param, value in kwargs.items():
            if param not in valid_params:
                self.get_logger().warning(f"忽略无效参数: {param}")
                continue
                
            if param in ["velocity", "position_tolerance", "angle_tolerance", "timeout", "lateral_threshold"]:
                if value <= 0:
                    self.get_logger().error(f"参数 {param} 必须大于0 (当前值: {value})")
                    continue
            elif param == "navigation_mode":  # 新增：导航模式验证
                if value.lower() not in ["all", "latest"]:
                    self.get_logger().error(f"无效的导航模式: {value}，必须是 'all' 或 'latest'")
                    continue
            elif param == "rotation_velocity":  # 新增：旋转速度验证
                if value <= 0:
                    self.get_logger().error(f"旋转速度必须大于0 (当前值: {value})")
                    continue
            
            setattr(self, param, value)
            # self.get_logger().info(f"参数更新: {param} = {value}")
    
    def odom_callback(self, msg):
        """更新机器人位置和方向"""
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.current_yaw = euler_from_quaternion(msg.pose.pose.orientation)
    
    def add_relative_waypoint(self, forward=0.0, left=0.0, yaw_offset=0.0):
        """
        添加相对当前位置的目标点
        
        参数:
            forward: 前方距离（正值为前方，负值为后方）
            left: 左侧距离（正值为左侧，负值为右侧）
            yaw_offset: 朝向偏移量（弧度）
        """
        # 计算相对偏移的绝对坐标
        x_offset = forward * math.cos(self.current_yaw) - left * math.sin(self.current_yaw)
        y_offset = forward * math.sin(self.current_yaw) + left * math.cos(self.current_yaw)
        
        # 计算绝对坐标
        abs_x = self.current_position[0] + x_offset
        abs_y = self.current_position[1] + y_offset
        abs_yaw = self.current_yaw + yaw_offset
        
        # 添加目标点
        self.waypoints.append((abs_x, abs_y, abs_yaw))
        self.get_logger().info(f"添加相对目标点: 前:{forward:.2f}m 左:{left:.2f}m 偏航:{yaw_offset:.2f}rad -> "
                              f"绝对位置({abs_x:.2f}, {abs_y:.2f}, {abs_yaw:.2f})")
    
    def clear_waypoints(self):
        """清除所有目标点"""
        self.waypoints = []
        # self.get_logger().info("已清除所有目标点")
    
    def set_waypoints(self, waypoints):
        """设置新的目标点序列"""
        self.waypoints = waypoints
        self.get_logger().info(f"已设置 {len(waypoints)} 个新目标点")
    
    def adjust_orientation(self, goal_yaw):
        """调整机器人朝向到目标方向（使用可配置的旋转速度）"""
        start_time = time.time()
        while self.active:
            angle_diff = goal_yaw - self.current_yaw
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # 归一化
            
            if abs(angle_diff) < self.angle_tolerance:
                self.cmd_pub.publish(create_twist())  # 停止
                # self.get_logger().info(f"朝向调整完成，目标朝向: {goal_yaw:.2f} 弧度")
                return True
            
            if time.time() - start_time > self.timeout:
                self.cmd_pub.publish(create_twist())  # 停止
                self.get_logger().error("朝向调整超时")
                return False
            
            # 使用配置的旋转速度
            angular_speed = self.rotation_velocity
            if abs(angle_diff) < 0.2:  # 接近目标时减速
                angular_speed *= 0.5
                
            if angle_diff > 0:
                self.cmd_pub.publish(create_twist(z=angular_speed))  # 左转
            else:
                self.cmd_pub.publish(create_twist(z=-angular_speed))  # 右转
            
            rclpy.spin_once(self, timeout_sec=0.1)
        
        return False

    def move_to_goal(self, goal_position, goal_yaw):
        """
        平滑移动到目标点（基于误差比例分配速度分量）
        
        参数:
            goal_position: 目标位置 (x, y)
            goal_yaw: 目标朝向
        """
        goal_x, goal_y = goal_position
        start_time = time.time()
        
        while self.active:
            # 获取当前位置和朝向
            current_x, current_y = self.current_position
            current_yaw = self.current_yaw
            
            # 计算世界坐标系下的误差
            dx = goal_x - current_x
            dy = goal_y - current_y
            distance_error = math.sqrt(dx**2 + dy**2)
            
            # 计算角度误差
            angle_diff = goal_yaw - current_yaw
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # 归一化
            
            # 检查是否到达目标位置
            if distance_error < self.position_tolerance and abs(angle_diff) < self.angle_tolerance:
                self.cmd_pub.publish(create_twist())  # 停止
                return True
                
            # 检查超时
            if time.time() - start_time > self.timeout:
                self.cmd_pub.publish(create_twist())
                self.get_logger().error("移动到目标点超时")
                return False
                
            # === 平滑合力运动控制 ===
            # 1. 计算世界坐标系下的方向向量
            if distance_error > 1e-5:  # 避免除零
                direction_x = dx / distance_error
                direction_y = dy / distance_error
            else:
                direction_x, direction_y = 0.0, 0.0
                
            # 2. 将方向向量转换到机器人坐标系
            vx_robot = direction_x * math.cos(current_yaw) + direction_y * math.sin(current_yaw)
            vy_robot = -direction_x * math.sin(current_yaw) + direction_y * math.cos(current_yaw)
            
            # 3. 计算速度大小（根据距离动态调整）
            # 基础速度
            base_speed = self.velocity
            
            # 接近目标时减速
            if distance_error < 0.5:
                speed_factor = max(0.1, distance_error / 0.5)  # 在0.5米内开始减速
                base_speed *= speed_factor
                
            # 4. 应用速度分量
            vx_robot *= base_speed
            vy_robot *= base_speed
            
            # === 旋转控制 ===
            # 计算角速度（P控制）
            angular_gain = 0.8  # P增益，可调整
            angular_z = angular_gain * angle_diff
            
            # 限制最大旋转速度
            max_angular_speed = self.rotation_velocity
            if abs(angular_z) > max_angular_speed:
                angular_z = max_angular_speed if angular_z > 0 else -max_angular_speed
            
            # === 发布速度指令 ===
            self.cmd_pub.publish(create_twist(x=vx_robot, y=vy_robot, z=angular_z))
            
            rclpy.spin_once(self, timeout_sec=0.05)  # 更快的控制循环
        
        return False
    
    def navigate_to_waypoint(self, waypoint):
        """导航到单个目标点"""
        goal_position = (waypoint[0], waypoint[1])
        goal_yaw = waypoint[2]
        self.target_yaw = goal_yaw
        # self.get_logger().info(f"导航到目标点: {goal_position}, 目标朝向: {goal_yaw:.2f} 弧度")
        
        # 使用平滑合力控制移动到目标点
        if not self.move_to_goal(goal_position, goal_yaw):
            return False
        
        # 微调朝向（可选，因为移动过程中已调整）
        # if not self.adjust_orientation(goal_yaw):
        #     return False
        
        return True
    
    def navigation_sequence(self):
        """执行整个导航序列"""
        self.navigation_complete = False
        self.navigation_failed = False
        
        try:
            # 根据导航模式确定要导航的目标点列表
            if self.navigation_mode == "latest" and self.waypoints:
                # 只导航到最新的一个点
                target_waypoints = [self.waypoints[-1]]
                # self.get_logger().info(f"导航模式: 只导航到最新添加的目标点")
            else:
                # 导航所有点
                target_waypoints = self.waypoints
            
            if not target_waypoints:
                self.get_logger().error("没有可导航的目标点")
                self.navigation_failed = True
                return
            
            for i, waypoint in enumerate(target_waypoints):
                # self.get_logger().info(f"开始导航到第 {i+1}/{len(target_waypoints)} 个目标点")
                if not self.navigate_to_waypoint(waypoint):
                    self.get_logger().error(f"导航到第 {i+1} 个目标点失败")
                    self.navigation_failed = True
                    return
            
            # if self.navigation_mode == "latest":
            #     self.get_logger().info("✅ 最新目标点导航完成!")
            # else:
            #     self.get_logger().info("✅ 所有目标点导航完成!")
                
            self.navigation_complete = True
        
        except Exception as e:
            self.get_logger().error(f"导航出错: {e}")
            self.navigation_failed = True
        finally:
            self.active = False
    
    def start_navigation(self):
        """非阻塞方式启动导航"""
        if self.active:
            self.get_logger().warn("导航已在运行中")
            return
        
        if not self.waypoints:
            self.get_logger().error("没有设置目标点，无法开始导航")
            return
        
        self.active = True
        self.navigation_thread = threading.Thread(target=self.navigation_sequence)
        self.navigation_thread.daemon = True
        self.navigation_thread.start()
        self.get_logger().info("导航已启动")
    
    def navigate(self):
        """阻塞方式执行导航"""
        if self.active:
            self.get_logger().warn("导航已在运行中")
            return
        
        if not self.waypoints:
            self.get_logger().error("没有设置目标点，无法开始导航")
            return
        
        self.active = True
        self.navigation_sequence()
    
    def stop_navigation(self):
        """停止导航"""
        self.active = False
        self.cmd_pub.publish(create_twist())  # 停止机器人
        self.get_logger().info("导航已停止")
    
    def is_active(self):
        """检查导航是否正在进行"""
        return self.active
    
    def is_complete(self):
        """检查导航是否成功完成"""
        return self.navigation_complete
    
    def has_failed(self):
        """检查导航是否失败"""
        return self.navigation_failed