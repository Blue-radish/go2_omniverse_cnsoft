#!/usr/bin/env python3
"""
obstacle_tracker.py - 点云追踪工具类（支持固定朝向版）

功能:
1. 订阅点云和里程计数据
2. 保持固定朝向（如X轴正方向）
3. 检测前方障碍物并计算中心点
4. 如果目标障碍物不居中，则左右平移使其居中（同时保持固定朝向）
5. 在障碍物中心点居中后向障碍物前进直到连续多次检测到距离达到阈值
6. 如果多次检测找不到目标，则扩大检测范围
7. 如果扩大检测范围后找到目标，则保持朝向移动使其中心点居中，然后恢复检测范围
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np
import time
import math

# ───────── 状态枚举 ──────────────────────────────────────────
class State:
    SEARCHING = 0      # 寻找障碍物
    ALIGNING = 1       # 旋转调整方向
    APPROACHING = 2    # 向障碍物前进
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

# ───────── 点云追踪工具类 ─────────────────────────────────────
class ObstacleTracker(Node):
    def __init__(self, 
                 context=None,
                 obstacle_threshold=0.5,
                 forward_speed=0.5,
                 turn_speed=0.8,
                 sideway_speed=0.3,  # 横向移动速度参数
                 fov_width=0.5,
                 min_points=5,
                 center_deadzone=0.12,
                 ground_threshold=0.1,
                 reached_count_threshold=3,
                 cmd_vel_topic="/robot0/cmd_vel",
                 point_cloud_topic="/robot0/point_cloud2",
                 odom_topic="/robot0/odom",
                 fov_enlarge_factor=2,  # FOV增大倍数
                 consecutive_miss_threshold=10,  # 连续丢失阈值
                 fixed_orientation=0,  # 固定朝向（弧度），None表示不固定
                 orientation_tolerance=0.05):  # 朝向容差（弧度）
        """
        初始化点云追踪器（支持固定朝向）
        
        参数:
            fixed_orientation: 机器人需要保持的固定朝向（弧度），None表示不固定
            orientation_tolerance: 朝向容差（弧度）
        """
        super().__init__("obstacle_tracker", context=context)
        
        # 参数验证
        self._validate_params(
            obstacle_threshold, forward_speed, turn_speed, fov_width, min_points,
            center_deadzone, ground_threshold, reached_count_threshold,
            fov_enlarge_factor, consecutive_miss_threshold, sideway_speed
        )
        
        # 存储参数
        self.obstacle_threshold = obstacle_threshold
        self.forward_speed = forward_speed
        self.turn_speed = turn_speed
        self.sideway_speed = sideway_speed
        self.fov_width = fov_width
        self.min_points = min_points
        self.center_deadzone = center_deadzone
        self.ground_threshold = ground_threshold
        self.reached_count_threshold = reached_count_threshold
        self.fov_enlarge_factor = fov_enlarge_factor
        self.consecutive_miss_threshold = consecutive_miss_threshold
        self.fixed_orientation = fixed_orientation  # 固定朝向
        self.orientation_tolerance = orientation_tolerance  # 朝向容差
        
        # 创建发布者
        self.pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        
        # 订阅点云和里程计话题
        self.point_cloud_sub = self.create_subscription(
            PointCloud2, point_cloud_topic, self.point_cloud_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10)
        
        # 机器人当前状态
        self.robot_position = Point()
        self.robot_yaw = 0.0
        
        # 障碍物追踪状态
        self.state = State.SEARCHING
        self.obstacle_detected = False
        self.obstacle_center = None
        self.obstacle_distance = float('inf')
        self.last_cmd_time = self.get_clock().now()
        self.current_cmd = make_twist()
        self.reached_count = 0          # 连续达到阈值的计数器
        self.active = False              # 追踪是否激活
        
        # 新增状态变量
        self.consecutive_misses = 0      # 连续未检测到障碍物的次数
        self.original_fov_width = fov_width  # 原始FOV宽度
        self.fov_enlarged = False        # 当前是否已增大FOV
        self.found_after_enlarge = False # 是否在增大FOV后找到目标
        self.orientation_correction_needed = False  # 是否需要纠正朝向
        
        # 启动控制循环
        self.timer = self.create_timer(0.1, self.control_loop)  # 10Hz控制循环

    def _validate_params(self,
                        obstacle_threshold,
                        forward_speed,
                        turn_speed,
                        fov_width,
                        min_points,
                        center_deadzone,
                        ground_threshold,
                        reached_count_threshold,
                        fov_enlarge_factor,
                        consecutive_miss_threshold,
                        sideway_speed):
        """验证参数是否有效"""
        names = [
            "obstacle_threshold", "forward_speed", "turn_speed", "fov_width",
            "min_points", "center_deadzone", "ground_threshold", "reached_count_threshold",
            "fov_enlarge_factor", "consecutive_miss_threshold", "sideway_speed"
        ]

        for name, value in zip(names, [
                obstacle_threshold, forward_speed, turn_speed, fov_width,
                min_points, center_deadzone, ground_threshold, reached_count_threshold,
                fov_enlarge_factor, consecutive_miss_threshold, sideway_speed]):
            if value < 0:  # 允许sideway_speed为0
                raise ValueError(f"参数 {name} 必须大于等于0 (当前值: {value})")

        if min_points < 1:
            raise ValueError(f"min_points 必须至少为1 (当前值: {min_points})")

        if reached_count_threshold < 1:
            raise ValueError(f"reached_count_threshold 必须至少为1 (当前值: {reached_count_threshold})")

        if consecutive_miss_threshold < 1:
            raise ValueError(f"consecutive_miss_threshold 必须至少为1 (当前值: {consecutive_miss_threshold})")

        if fov_enlarge_factor <= 1.0:
            raise ValueError(f"fov_enlarge_factor 必须大于1.0 (当前值: {fov_enlarge_factor})")

    def update_parameters(self, **kwargs):
        """运行时更新参数"""
        valid_params = [
            "obstacle_threshold", "forward_speed", "turn_speed", "fov_width",
            "min_points", "center_deadzone", "ground_threshold", "reached_count_threshold",
            "fov_enlarge_factor", "consecutive_miss_threshold", "sideway_speed",
            "fixed_orientation", "orientation_tolerance"
        ]
        
        for param, value in kwargs.items():
            if param not in valid_params:
                self.get_logger().warning(f"忽略无效参数: {param}")
                continue
                
            if value < 0 and param != "fixed_orientation":  # fixed_orientation可以为负
                self.get_logger().error(f"参数 {param} 必须大于等于0 (当前值: {value})")
                continue
                
            if param == "min_points" and value < 1:
                self.get_logger().error(f"min_points 必须至少为1 (当前值: {value})")
                continue
                
            if param == "reached_count_threshold" and value < 1:
                self.get_logger().error(f"reached_count_threshold 必须至少为1 (当前值: {value})")
                continue
                
            if param == "consecutive_miss_threshold" and value < 1:
                self.get_logger().error(f"consecutive_miss_threshold 必须至少为1 (当前值: {value})")
                continue
                
            if param == "fov_enlarge_factor" and value <= 1.0:
                self.get_logger().error(f"fov_enlarge_factor 必须大于1.0 (当前值: {value})")
                continue
                
            setattr(self, param, value)
            # self.get_logger().info(f"参数更新: {param} = {value}")
            
            # 如果是固定朝向参数更新，需要重置状态
            if param == "fixed_orientation":
                self.orientation_correction_needed = True

    def odom_callback(self, msg):
        """更新机器人位置和方向"""
        self.robot_position = msg.pose.pose.position
        
        # 从四元数计算偏航角（yaw）
        orientation = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y**2 + orientation.z**2)
        self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # 检查是否需要纠正朝向
        if self.fixed_orientation is not None:
            angle_diff = self._normalize_angle(self.fixed_orientation - self.robot_yaw)
            if abs(angle_diff) > self.orientation_tolerance:
                self.orientation_correction_needed = True
                # self.get_logger().debug(f"朝向偏差: {math.degrees(angle_diff):.1f}度", 
                #                       throttle_duration_sec=1.0)
        
    def _normalize_angle(self, angle):
        """将角度归一化到[-π, π]区间"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
        
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
                
                # 重置连续丢失计数
                self.consecutive_misses = 0
                
                # self.get_logger().debug(
                #     f"检测到障碍物: 距离={min_distance:.2f}米", 
                #     throttle_duration_sec=1.0)
            else:
                self.obstacle_detected = False
                self.obstacle_center = None
                self.consecutive_misses += 1
                
        except Exception as e:
            self.get_logger().error(f"处理点云数据时出错: {str(e)}")
            self.obstacle_detected = False
            self.obstacle_center = None
            self.consecutive_misses += 1

    def control_loop(self):
        """基于状态机的控制循环"""
        if not self.active:
            return
            
        # 检查是否需要纠正朝向（优先级最高）
        if self.fixed_orientation is not None and self.orientation_correction_needed:
            angle_diff = self._normalize_angle(self.fixed_orientation - self.robot_yaw)
            if abs(angle_diff) > self.orientation_tolerance:
                # 保存当前状态，以便纠正后返回
                prev_state = self.state
                
                # 进入朝向纠正状态
                self.state = State.ORIENTING
                # self.get_logger().info(f"朝向偏离，纠正中... (偏差: {math.degrees(angle_diff):.1f}度)")
                
                # 计算旋转方向
                turn_direction = 1.0 if angle_diff > 0 else -1.0
                self.publish_continuous(make_twist(z=turn_direction * self.turn_speed))
                
                # 等待纠正完成
                return
            else:
                # 朝向已正确，重置标志
                self.orientation_correction_needed = False
                # self.get_logger().info("朝向已纠正")
                # 回到之前的状态
                
        # 检查连续丢失次数
        if self.consecutive_misses >= self.consecutive_miss_threshold:
            if not self.fov_enlarged:
                # 第一次连续丢失超过阈值，增大FOV
                self.original_fov_width = self.fov_width
                self.fov_width *= self.fov_enlarge_factor
                self.fov_enlarged = True
                self.get_logger().warn(
                    f"连续{self.consecutive_miss_threshold}次未检测到目标，增大FOV至{self.fov_width:.2f}米")
            else:
                # 已经增大FOV后仍然连续丢失，停止追踪
                self.get_logger().error(
                    f"增大FOV后仍然连续{self.consecutive_miss_threshold}次未检测到目标，停止追踪")
                self.stop_tracking()
                return
        
        # 状态机处理
        if self.state == State.SEARCHING:
            if self.obstacle_detected:
                if self.obstacle_distance <= self.obstacle_threshold:
                    # 连续达到阈值计数
                    self.reached_count += 1
                    if self.reached_count >= self.reached_count_threshold:
                        self.state = State.REACHED
                        # self.get_logger().info("连续达到阈值距离，停止。")
                    else:
                        # self.get_logger().info(f"达到阈值距离 ({self.reached_count}/{self.reached_count_threshold})")
                        pass
                else:
                    # 检查是否因增大FOV而找到目标
                    if self.fov_enlarged:
                        self.found_after_enlarge = True
                        # 根据横向速度决定行为
                        if self.sideway_speed > 0:
                            self.state = State.MOVING_SIDEWAYS
                            # self.get_logger().info("增大FOV后检测到目标，开始横向移动居中")
                        else:
                            self.state = State.ALIGNING
                            # self.get_logger().info("增大FOV后检测到目标，横向速度为0，直接调整方向")
                    else:
                        self.found_after_enlarge = False
                        # 如果设置了固定朝向，优先使用横向移动
                        if self.fixed_orientation is not None and self.sideway_speed > 0:
                            self.state = State.MOVING_SIDEWAYS
                            # self.get_logger().info("检测到障碍物，开始横向移动使其居中。")
                        else:
                            self.state = State.ALIGNING
                            # self.get_logger().info("检测到障碍物，开始旋转调整方向。")
            else:
                # 没有检测到障碍物，停止
                self.publish_once(make_twist())
                # self.get_logger().info("视野内未检测到障碍物，停止。", throttle_duration_sec=1.0)
                self.reached_count = 0  # 重置计数
        
        elif self.state == State.ALIGNING:
            if not self.obstacle_detected:
                self.state = State.SEARCHING
                # self.get_logger().info("调整过程中障碍物消失，返回搜索状态。")
                self.reached_count = 0
            elif self.obstacle_distance <= self.obstacle_threshold:
                # 连续达到阈值计数
                self.reached_count += 1
                if self.reached_count >= self.reached_count_threshold:
                    self.state = State.REACHED
                    self.get_logger().info("调整过程中达到阈值距离，停止。")
                else:
                    self.get_logger().info(f"达到阈值距离 ({self.reached_count}/{self.reached_count_threshold})")
            else:
                # 计算障碍物在视野中的横向偏移
                center_y = self.obstacle_center[1]
                
                # 检查障碍物是否已居中
                if abs(center_y) < self.center_deadzone:
                    # 如果之前是增大FOV后找到的目标，现在恢复FOV
                    if self.found_after_enlarge and self.fov_enlarged:
                        self._restore_fov()
                    
                    self.state = State.APPROACHING
                    # self.get_logger().info("障碍物已居中，开始前进。")
                    self.reached_count = 0  # 重置计数
                else:
                    # 障碍物未居中，原地旋转调整方向
                    turn_speed = np.sign(center_y) * self.turn_speed
                    self.publish_continuous(make_twist(z=turn_speed))
                    
                    # 记录旋转方向
                    direction = "向左" if turn_speed < 0 else "向右"
                    # self.get_logger().info(
                    #     f"障碍物偏移: {center_y:.2f}米, 原地{direction}旋转中", 
                    #     throttle_duration_sec=0.5)
        
        elif self.state == State.APPROACHING:
            if not self.obstacle_detected:
                self.state = State.SEARCHING
                # self.get_logger().info("前进过程中障碍物消失，返回搜索状态。")
                self.reached_count = 0
            elif self.obstacle_distance <= self.obstacle_threshold:
                # 连续达到阈值计数
                self.reached_count += 1
                if self.reached_count >= self.reached_count_threshold:
                    self.state = State.REACHED
                    # self.get_logger().info("已到达障碍物阈值距离，停止。")
                else:
                    # self.get_logger().info(f"达到阈值距离 ({self.reached_count}/{self.reached_count_threshold})")
                    pass
            else:
                # 检查障碍物是否仍然居中
                center_y = self.obstacle_center[1]
                if abs(center_y) > self.center_deadzone:
                    # 如果设置了固定朝向，优先使用横向移动
                    if self.fixed_orientation is not None and self.sideway_speed > 0:
                        self.state = State.MOVING_SIDEWAYS
                        # self.get_logger().info("前进过程中障碍物偏移，横向移动重新居中。")
                    else:
                        self.state = State.ALIGNING
                        # self.get_logger().info("前进过程中障碍物偏移，重新调整方向。")
                    self.reached_count = 0
                else:
                    # 持续前进
                    self.publish_continuous(make_twist(x=self.forward_speed))
                    # self.get_logger().info("向障碍物前进中...", throttle_duration_sec=0.5)
        
        elif self.state == State.REACHED:
            # 停止状态
            self.publish_once(make_twist())
            # 追踪完成，重置状态
            self.active = False
            self.state = State.SEARCHING
            
        elif self.state == State.MOVING_SIDEWAYS:
            if not self.obstacle_detected:
                self.state = State.SEARCHING
                # self.get_logger().info("横向移动过程中障碍物消失，返回搜索状态。")
            else:
                center_y = self.obstacle_center[1]
                if abs(center_y) < self.center_deadzone:
                    # 居中完成，恢复FOV并进入前进状态
                    if self.found_after_enlarge and self.fov_enlarged:
                        self._restore_fov()
                    
                    self.state = State.APPROACHING
                    # self.get_logger().info("横向移动完成，目标已居中，开始前进。")
                else:
                    # 计算横向移动方向（保持固定朝向）
                    move_direction = 1 if center_y > 0 else -1  # 正中心y表示左侧，所以向右移动
                    self.publish_continuous(make_twist(y=move_direction * self.sideway_speed))
                    
                    direction = "向右" if move_direction > 0 else "向左"
                    # self.get_logger().info(
                    #     f"横向移动: {direction} {self.sideway_speed:.2f}米/秒, 偏移: {center_y:.2f}米", 
                    #     throttle_duration_sec=0.5)
        
        elif self.state == State.ORIENTING:
            # 检查是否已经完成朝向纠正
            angle_diff = self._normalize_angle(self.fixed_orientation - self.robot_yaw)
            if abs(angle_diff) <= self.orientation_tolerance:
                # 纠正完成，返回之前的状态
                self.orientation_correction_needed = False
                self.state = State.SEARCHING  # 默认回到搜索状态
                # self.get_logger().info("朝向纠正完成，返回追踪状态。")
            else:
                # 继续纠正朝向
                turn_direction = 1.0 if angle_diff > 0 else -1.0
                self.publish_continuous(make_twist(z=turn_direction * self.turn_speed))

    def _restore_fov(self):
        """恢复原始FOV宽度"""
        if self.fov_enlarged:
            self.fov_width = self.original_fov_width
            self.fov_enlarged = False
            self.found_after_enlarge = False
            self.get_logger().info(f"恢复FOV宽度至{self.fov_width:.2f}米")

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
        
    def start_tracking(self):
        """开始追踪过程"""
        self.active = True
        self.state = State.SEARCHING
        self.reached_count = 0
        self.consecutive_misses = 0
        self.fov_enlarged = False
        self.found_after_enlarge = False
        
        # 如果设置了固定朝向，确保初始方向正确
        if self.fixed_orientation is not None:
            angle_diff = self._normalize_angle(self.fixed_orientation - self.robot_yaw)
            if abs(angle_diff) > self.orientation_tolerance:
                self.state = State.ORIENTING
                # self.get_logger().info("初始朝向不正确，先进行纠正...")
        
        # self.get_logger().info("开始追踪障碍物")
        
    def stop_tracking(self):
        """停止追踪过程"""
        self.active = False
        self.stop_robot()
        self.get_logger().info("停止追踪")
        
    def is_tracking_active(self):
        """检查是否正在追踪"""
        return self.active
        
    def has_reached_target(self):
        """检查是否达到目标"""
        return self.state == State.REACHED
        
    def track_until_reached(self):
        """开始追踪直到连续达到障碍物距离阈值（阻塞式）"""
        self.start_tracking()
        
        # 重置状态并开始追踪
        while rclpy.ok() and self.active:
            rclpy.spin_once(self, timeout_sec=0.1)
            
        # 返回是否因为达到目标而停止
        return self.state == State.REACHED