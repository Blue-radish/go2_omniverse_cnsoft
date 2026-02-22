#!/usr/bin/env python3
"""
go2_motion_demo.py  ·  Unitree Go2 机器人的固定运动序列

向 /robot0/cmd_vel 发送 Twist 消息：
1. 向前移动 2 秒
2. 向左移动 1 秒
3. 向前移动 3 秒
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

# ───────── Twist 辅助函数 ───────────────────────────────────────────
def make_twist(x=0.0, y=0.0, z=0.0):
    """
    创建一个 Twist 消息，包含线速度和角速度。
    :param x: 线速度在 x 方向（前进/后退）
    :param y: 线速度在 y 方向（侧移）
    :param z: 角速度在 z 方向（旋转）
    :return: Twist 消息
    """
    t = Twist()
    t.linear.x = x
    t.linear.y = y
    t.angular.z = z
    return t

# ───────── ROS 2 示例节点 ────────────────────────────────────────
class Go2MotionDemo(Node):
    def __init__(self):
        """
        初始化 ROS 2 节点，创建一个发布者，发布到 /robot0/cmd_vel 话题。
        """
        super().__init__("go2_motion_demo")
        self.pub = self.create_publisher(Twist, "/robot0/cmd_vel", 10)
        time.sleep(1.0)  # 等待发布者连接

        self.get_logger().info("开始固定 Go2 运动演示...")
        self.run_demo()

    def run_demo(self):
        """
        执行运动序列：
        1. 向前移动 2 秒
        2. 停止 1 秒
        3. 向左移动 1 秒
        4. 停止 1 秒
        5. 向前移动 3 秒
        6. 停止
        """
        # 向前移动 1 秒
        self.get_logger().info("向前移动 1 秒")
        self.pub.publish(make_twist(x=1.0))
        time.sleep(1.0)

        # 停止 1 秒
        self.pub.publish(make_twist())
        time.sleep(1.0)

        # 向左移动 2 秒
        self.get_logger().info("向左移动 2 秒")
        self.pub.publish(make_twist(y=2.0))
        time.sleep(2.0)

        # 停止 1 秒
        self.pub.publish(make_twist())
        time.sleep(1.0)

        # 向前移动 3 秒
        self.get_logger().info("向前移动 3 秒")
        self.pub.publish(make_twist(x=1.0))
        time.sleep(3.0)

        # 停止
        self.pub.publish(make_twist())
        self.get_logger().info("✅ 运动演示完成。")

# ───────── 入口点 ────────────────────────────────────────────
def main(args=None):
    """
    程序入口点，初始化 ROS 2，创建节点并运行一次运动演示。
    """
    rclpy.init(args=args)
    demo = Go2MotionDemo()
    rclpy.spin_once(demo, timeout_sec=0)  # 只运行一次
    demo.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
