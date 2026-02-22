#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class RobotOdomListener(Node):
    def __init__(self):
        super().__init__('robot_odom_listener')
        # 订阅/robot0/odom话题
        self.subscription = self.create_subscription(
            Odometry,
            '/robot0/odom',
            self.odom_callback,
            10)
        self.subscription  # 防止未使用的变量警告

    def odom_callback(self, msg):
        # 提取机器人的位置信息
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        self.get_logger().info(f"机器人当前位置: x = {x:.2f}, y = {y:.2f}, z = {z:.2f}")

def main(args=None):
    rclpy.init(args=args)
    robot_odom_listener = RobotOdomListener()
    rclpy.spin(robot_odom_listener)
    robot_odom_listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()