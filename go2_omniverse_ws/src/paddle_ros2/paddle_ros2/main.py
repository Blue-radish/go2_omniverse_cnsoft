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
        super().__init__("main")
        
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
            "\n"
            # f"动作完成: {action_name}\n"
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
        # self.get_logger().info(f"收到方向指令: {direction} (总计: {self.direction_counter})")

    def execute_motion_with_ocr(self, duration, twist_cmd, description):
        """执行指定动作同时记录OCR方向指令"""
        self.get_logger().info(f"OCR:开始 OCR并{description}")
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
        self.get_logger().info(f"OCR:结束 OCR并{description}")
        
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
        self.get_logger().info(f"OCR:最终选择方向 {direction}  (出现次数 {count})")
        return direction

    def execute_motion(self, duration, twist_cmd, description):
        """执行指定动作（不带OCR记录）"""
        self.get_logger().info(f"{description} 开始")
        start_time = time.time()
        
        while time.time() - start_time < duration and rclpy.ok():
            self.cmd_pub.publish(twist_cmd)
            
            # 处理ROS回调（接收里程计）
            rclpy.spin_once(self, timeout_sec=0.05)
            
            time.sleep(0.05)  # 控制循环频率
        
        # 发送停止命令
        self.cmd_pub.publish(make_twist())
        self.get_logger().info(f"{description} 结束")

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
            "后退"
        )
        demo.print_position("OCR:后退并OCR识别")  # 打印位置
        
        # 如果没有识别到方向，使用默认值（左）
        if not direction:
            direction = "left"
        direction = "right"
        time.sleep(3.0)

        demo.get_logger().info("避障:点云追踪 开始")           # 点云追踪到避障障碍物前
        tracker.update_parameters(obstacle_threshold=0.53,forward_speed=0.55)
        tracker.track_until_reached()
        demo.get_logger().info("避障:点云追踪 结束")
        demo.print_position("避障:点云追踪")  # 打印位置
        time.sleep(3.0)

        # 根据OCR结果执行不同的越障动作
        if direction == "right":
            navigator.waypoints.append((0.92, -0.4 ,0.0)) # 目标点绝对坐标
            navigator.update_parameters(velocity=0.5)
            navigator.navigate()
            demo.get_logger().info("避障:right避障")
            demo.print_position("避障:right避障")  # 打印位置
            time.sleep(1.0)
            navigator.waypoints.append((1.85, -0.4 ,0.0)) # 目标点绝对坐标
            navigator.update_parameters(velocity=0.5)
            navigator.navigate()
            # demo.print_position("避障:right避障")  # 打印位置
            time.sleep(1.0)
            navigator.update_parameters(velocity=0.5)
            navigator.waypoints.append((2.0,-0.1,0.0))
            navigator.navigate()
            # demo.print_position("避障:right避障")  # 打印位置
            time.sleep(1.0)
        else:  # 默认向左
            navigator.waypoints.append((0.9, 0.5 ,0.0)) # 目标点绝对坐标
            navigator.update_parameters(velocity=0.6, timeout=15)
            navigator.navigate()
            demo.get_logger().info("避障:left避障")
            demo.print_position("避障:left避障")  # 打印位置
            time.sleep(1.0)

            navigator.waypoints.append((1.85,0.5,0.0))
            navigator.update_parameters(velocity=0.5)
            navigator.navigate()
            # demo.print_position("避障:left避障")  # 打印位置
            time.sleep(1.0)

            navigator.update_parameters(velocity=0.5)
            navigator.waypoints.append((2.0,0.1,0.0))
            navigator.navigate()
            # demo.print_position("避障:left避障")  # 打印位置
            time.sleep(1.0)
        
        align_and_detect(context=demo.context)            # 拍照识别图片1
        demo.print_position("图片1:识别")  # 打印位置
        time.sleep(1.0)
        
        demo.execute_motion(7, make_twist(x=0.5,z=1.0), "前往图片2:跳过越障任务") # 转弯,跳过越障任务
        demo.print_position("图片2:跳过越障任务")  # 打印位置
        
        demo.execute_motion(12, make_twist(x=0.5,z=1.0), "前往图片2:逆时针转弯 180度")
        demo.print_position("图片2:逆时针转弯180度")  # 打印位置
        time.sleep(1.0)

        navigator.clear_waypoints()
        navigator.waypoints.append((1.8, 1.6 ,math.pi)) # 目标点绝对坐标
        navigator.update_parameters(velocity=0.7, rotation_velocity=0.5) # 更新参数 先x调整后y调整
        demo.get_logger().info("图片2:走到图片2前")
        navigator.navigate()
        demo.print_position("图片2:走到图片2前")  # 打印位置
        time.sleep(1.0)

        align_and_detect(context=demo.context)           # 拍照识别图片2
        demo.print_position("图片2:识别")  # 打印位置
        time.sleep(1.0)

        demo.get_logger().info("图片2:点云追踪 开始")          # 靠近,准备转弯，走向图片3
        tracker.update_parameters(obstacle_threshold=1.0, forward_speed=0.8,sideway_speed=0.3, fov_width=0.5, fixed_orientation=math.pi)
        tracker.track_until_reached()
        demo.get_logger().info("图片2:点云追踪 结束")
        demo.print_position("图片2:点云追踪")  # 打印位置
        time.sleep(1.0)

        demo.execute_motion(8, make_twist(x=0.5,z=-1.0), "图片3:顺时针90度") # 转弯, 准备前往识别任务3
        demo.print_position("图片3:顺时针90度")  # 打印位置
        time.sleep(1.0)

        demo.execute_motion(6,make_twist(x=-0.6),"图片3:后退") # 后退保证能够识别到图片3整个图片
        demo.print_position("图片3:后退") 
        time.sleep(1.0)

        demo.get_logger().info("图片3:点云追踪 开始")
        tracker.update_parameters(obstacle_threshold=1.5,fixed_orientation=math.pi / 2) #追踪调整狗的头位，保证正对图片3
        tracker.track_until_reached()
        demo.get_logger().info("图片3:点云追踪 结束")
        demo.print_position("图片3:点云追踪")  # 打印位置
        time.sleep(1.0)

        align_and_detect(context=demo.context)           # 拍照识别图片3
        demo.print_position("图片3:识别")  # 打印位置
        time.sleep(1.0)

        demo.execute_motion(10, make_twist(x=0.5,z=-1.0), "终点:顺时针90度") #转弯, 准备前往终点位置
        demo.print_position("终点:顺时针90度")  # 打印位置
        time.sleep(1.0)

        demo.get_logger().info("终点:一次定点 开始")
        navigator.clear_waypoints()
        navigator.waypoints.append((1.8, 2.65, 0.0)) # 目标点绝对坐标
        navigator.update_parameters(velocity=0.7, rotation_velocity=0.5,timeout=15)
        navigator.navigate()
        demo.get_logger().info("终点:一次定点 结束")
        demo.print_position("终点:一次定点")  # 打印位置
        time.sleep(1.0)

        demo.get_logger().info("终点:二次追踪 开始")
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
        demo.get_logger().info("终点:二次追踪 结束") # 一次定点后，在进入终点区域时，使用虚拟障碍点云追踪方案，精确到终点
        demo.print_position("终点:二次追踪")  # 打印位置
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