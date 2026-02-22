#!/usr/bin/env python3
"""
main_copy.py ocr识别功能测试程序

组合策略
"""

import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import time
import collections
from tools.obstacle_tracker import ObstacleTracker  # 导入点云追踪工具
from tools.align_and_detect import align_and_detect  # 导入方向调整与目标检测功能
from tools.waypoint_navigator import WaypointNavigator
from tools.point_tracker import PointTracker

# ───────── Twist 助手函数 ───────────────────────────────────────────
def make_twist(x=0.0, y=0.0, z=0.0):
    t = Twist()
    t.linear.x = x
    t.linear.y = y
    t.angular.z = z  # 绕Z轴的偏航角(yaw)使用传入的值
    return t

# 四元数转欧拉角 (yaw)
def quaternion_to_yaw(q):
    # 计算偏航角 (yaw)
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw

# ───────── 主节点 ─────────────────────────────────────────────────
class MainDemo(Node):
    def __init__(self):
        super().__init__("main_demo")
        self.cmd_pub = self.create_publisher(Twist, "/robot0/cmd_vel", 10)
        self.direction_sub = self.create_subscription(
            String, "/robot0/direction_cmd", self.direction_callback, 10
        )
        
        # 添加里程计订阅
        self.odom_sub = self.create_subscription(
            Odometry, "/robot0/odom", self.odom_callback, 10
        )
        self.current_position = [0.0, 0.0, 0.0]  # [x, y, yaw]
        self.latest_odom = None  # 存储最新的里程计消息
        
        # OCR相关变量
        self.direction_counter = collections.Counter()  # 用于计数方向指令
        self.direction_queue = []  # 存储接收到的方向指令
        self.latest_direction = None  # 最新接收到的方向指令
        self.ocr_received = False
        self.ocr_timeout = 5.0  # 超时时间(秒)
        
        time.sleep(1.0)  # 等待发布器连接

    def odom_callback(self, msg: Odometry):
        """处理里程计回调，更新当前位置和朝向"""
        self.latest_odom = msg
        
        # 获取位置
        self.current_position[0] = msg.pose.pose.position.x
        self.current_position[1] = msg.pose.pose.position.y
        
        # 获取朝向 (yaw)
        q = msg.pose.pose.orientation
        self.current_position[2] = quaternion_to_yaw(q)

    def print_position(self, action_name):
        """强制处理最新里程计数据并打印位置"""
        # 处理最新的里程计消息
        if self.latest_odom:
            self.odom_callback(self.latest_odom)
        
        # 确保位置信息是最新的
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.01)
        
        x, y, yaw = self.current_position
        # 将弧度转换为角度
        yaw_deg = math.degrees(yaw)
        self.get_logger().info(
            f"动作完成: {action_name}\n"
            f"当前位置: x={x:.2f}m, y={y:.2f}m\n"
            f"当前朝向: yaw={yaw_deg:.1f}°"
        )

    def direction_callback(self, msg: String):
        """处理方向指令回调"""
        direction = msg.data
        self.latest_direction = direction
        self.direction_queue.append(direction)
        self.direction_counter[direction] += 1
        self.ocr_received = True
        self.get_logger().info(f"收到方向指令: {direction} (总计: {self.direction_counter})")

    def execute_motion_with_ocr(self, duration, twist_cmd, description):
        """执行指定动作同时记录OCR方向指令"""
        self.get_logger().info(f"开始执行动作并记录OCR: {description}")
        self.direction_counter.clear()  # 清空前一次记录的指令
        self.direction_queue.clear()
        self.ocr_received = False
        
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time and rclpy.ok():
            # 发布运动命令
            self.cmd_pub.publish(twist_cmd)
            
            # 处理ROS回调（接收OCR指令和里程计）
            rclpy.spin_once(self, timeout_sec=0.05)
            
            # 控制循环频率
            time.sleep(0.05)
        
        # 发送停止命令
        self.cmd_pub.publish(make_twist())
        self.get_logger().info(f"动作完成: {description}")
        
        # 返回收集到的方向指令
        return self.get_most_common_direction()

    def get_most_common_direction(self):
        """获取出现次数最多的方向指令"""
        if not self.direction_counter:
            self.get_logger().warn("未收到任何方向指令")
            return None
        
        # 找出出现次数最多的方向
        most_common = self.direction_counter.most_common(1)
        direction, count = most_common[0]
        self.get_logger().info(f"最终选择方向: {direction} (出现次数: {count})")
        return direction

    def execute_motion(self, duration, twist_cmd, description):
        """执行指定动作（不带OCR记录）"""
        self.get_logger().info(f"执行动作: {description}")
        start_time = time.time()
        
        while time.time() - start_time < duration and rclpy.ok():
            self.cmd_pub.publish(twist_cmd)
            
            # 处理ROS回调（接收里程计）
            rclpy.spin_once(self, timeout_sec=0.05)
            
            time.sleep(0.05)  # 控制循环频率
        
        # 发送停止命令
        self.cmd_pub.publish(make_twist())
        self.get_logger().info(f"动作完成: {description}")

# ───────── 主函数 ────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    demo = MainDemo()
    
    tracker = ObstacleTracker(context=demo.context)
    navigator = WaypointNavigator(context=demo.context)
    point_tracker = PointTracker(context=demo.context)

    try:
        direction = demo.execute_motion_with_ocr(
            16, 
            make_twist(x=-0.5), 
            "后退并OCR识别"
        )
        demo.print_position("后退并OCR识别")  # 打印位置
        
        # 如果没有识别到方向，使用默认值（左）
        if not direction:
            direction = "left"
        
        time.sleep(1.5)

        demo.get_logger().info("启动点云追踪...")           # 点云追踪到避障障碍物前
        tracker.update_parameters(obstacle_threshold=0.51,forward_speed=0.65)
        tracker.track_until_reached()
        demo.get_logger().info("点云追踪结束")
        demo.print_position("点云追踪")  # 打印位置
        time.sleep(1.5)

        # direction="left"

        # 根据OCR结果执行不同的越障动作
        if direction == "right":
            navigator.waypoints.append((0.9, -0.4 ,0.0)) # 绝对坐标方式给出运动
            navigator.update_parameters(movement_axis_order='xy',velocity=0.6)
            navigator.navigate()
            demo.print_position("right避障-1")  # 打印位置
            time.sleep(1.0)
            navigator.waypoints.append((1.8, -0.4 ,0.0)) # 绝对坐标的方式给出运动
            navigator.navigate()
            demo.print_position("right避障-2")  # 打印位置
            time.sleep(1.0)
            navigator.update_parameters(velocity=0.5)
            navigator.waypoints.append((1.9,-0.1,0.0))
            navigator.navigate()
            demo.print_position("right避障-3")  # 打印位置
            time.sleep(1.0)
        else:  # 默认向左
            navigator.waypoints.append((0.9, 0.5 ,0.0)) # 绝对坐标方式给出运动
            navigator.update_parameters(movement_axis_order='xy',velocity=0.6)
            navigator.navigate()
            demo.print_position("left避障-1 左移")  # 打印位置
            time.sleep(1.0)

            # 创建目标点
            target_point = Point()
            target_point.x = 2.3
            target_point.y = 0.5
            # 一行调用追踪功能
            point_tracker.track_to_point(
                target_point=target_point,
                target_threshold=0.56,
                forward_speed=0.6,
                sideway_speed=0.3,
                fixed_orientation=0  # 保持朝向
            )
            demo.print_position("left避障-2 前进")  # 打印位置
            time.sleep(1.0)

            navigator.update_parameters(movement_axis_order='yx',velocity=0.5)
            navigator.waypoints.append((2.1,0.1,0.0))
            navigator.navigate()
            demo.print_position("left避障-3 右移")  # 打印位置
            time.sleep(1.0)

            # navigator.waypoints.append((0.9, 0.5 ,0.0)) # 绝对坐标方式给出运动
            # navigator.update_parameters(velocity=0.7)
            # navigator.navigate()
            # demo.print_position("left避障-1")  # 打印位置
            # time.sleep(1.0)
            # navigator.waypoints.append((1.85, 0.5 ,0.0)) # 绝对坐标的方式给出运动
            # navigator.navigate()
            # demo.print_position("left避障-2")  # 打印位置
            # time.sleep(1.0)
            # navigator.waypoints.append((2.0,0.1,0.0))
            # navigator.navigate()
            # demo.print_position("left避障-3")  # 打印位置
            # time.sleep(1.0)
        
        align_and_detect(context=demo.context)            # 识别任务1图片
        demo.print_position("识别任务1")  # 打印位置
        time.sleep(1.0)
        
        demo.execute_motion(7, make_twist(x=0.5,z=1.0), "逆时针转弯 180度-1") # 转弯,跳过越障任务,但识别越障图片
        demo.print_position("逆时针转弯 180度-1")  # 打印位置
        align_and_detect(context=demo.context, enable_alignment=False)
        demo.print_position("识别越障图片")  # 打印位置
        
        demo.execute_motion(12, make_twist(x=0.5,z=1.0), "逆时针转弯 180度-2")
        demo.print_position("逆时针转弯 180度-2")  # 打印位置
        time.sleep(1.0)

        navigator.clear_waypoints()
        navigator.waypoints.append((1.9, 1.6 ,math.pi)) # 绝对坐标方式给出目标点位
        navigator.update_parameters(velocity=0.7, movement_axis_order='xy',rotation_velocity=0.5)
        navigator.navigate()
        demo.print_position("汇入前往图片2路线")  # 打印位置
        time.sleep(1.0)

        # demo.get_logger().info("启动点云追踪...")    # 追踪前进到识别任务2图片前
        # tracker.update_parameters(obstacle_threshold=1.5, forward_speed=0.8,sideway_speed=0.3, fov_width=0.5, fixed_orientation=math.pi)
        # tracker.track_until_reached()
        # demo.get_logger().info("点云追踪结束")
        # demo.print_position("点云追踪")  # 打印位置
        # time.sleep(1.0)

        align_and_detect(context=demo.context)           # 识别任务2图片
        demo.print_position("识别任务2")  # 打印位置
        time.sleep(1.0)

        demo.get_logger().info("启动点云追踪...")          # 靠近识别2图片,转弯准备
        tracker.update_parameters(obstacle_threshold=1.0, forward_speed=0.8,sideway_speed=0.3, fov_width=0.5, fixed_orientation=math.pi)
        tracker.track_until_reached()
        demo.get_logger().info("点云追踪结束")
        demo.print_position("点云追踪")  # 打印位置
        time.sleep(1.0)

        demo.execute_motion(8, make_twist(x=0.5,z=-1.0), "顺时针 90度") #转弯, 准备前往识别任务3
        demo.print_position("顺时针 90度")  # 打印位置
        time.sleep(1.0)

        demo.execute_motion(5,make_twist(x=-0.6),"后退")
        demo.print_position("后退准备识别图片3")
        time.sleep(1.0)

        demo.get_logger().info("启动点云追踪...")
        tracker.update_parameters(obstacle_threshold=1.5,fixed_orientation=math.pi / 2) #追踪前进到识别任务3图片前
        tracker.track_until_reached()
        demo.get_logger().info("点云追踪结束")
        demo.print_position("点云追踪")  # 打印位置
        time.sleep(1.0)

        align_and_detect(context=demo.context)           # 识别任务3图片
        demo.print_position("识别任务3")  # 打印位置
        time.sleep(1.0)

        demo.execute_motion(10, make_twist(x=0.5,z=-1.0), "顺时针 90度") #转弯, 准备前往终点位置
        demo.print_position("顺时针 90度")  # 打印位置
        time.sleep(1.0)
        

        navigator.clear_waypoints()
        navigator.waypoints.append((1.8, 2.65, 0.0)) # 绝对坐标方式给出目标点位
        navigator.update_parameters(velocity=0.7, movement_axis_order='xy',rotation_velocity=0.5)
        navigator.navigate()
        demo.print_position("前往终点-1")  # 打印位置
        time.sleep(1.0)

        # navigator.waypoints.append((3.0, 2.7, 0.0)) # 绝对坐标方式给出目标点位
        # navigator.update_parameters(velocity=0.4, rotation_velocity=0.5)
        # navigator.navigate()
        # demo.print_position("前往终点-2")  # 打印位置
        # time.sleep(1.0)

        # 创建目标点
        target_point = Point()
        target_point.x = 3.5
        target_point.y = 2.65
        # 一行调用追踪功能
        point_tracker.track_to_point(
            target_point=target_point,
            target_threshold=0.5,
            forward_speed=0.5,
            sideway_speed=0.3,
            fixed_orientation=0  # 保持朝向
        )
        demo.print_position("前往终点-2")  # 打印位置
        time.sleep(1.0)


        demo.get_logger().info("✅ 演示程序完成")
        
    finally:
        # 确保机器人停止
        demo.cmd_pub.publish(make_twist())
        tracker.destroy_node()
        demo.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()