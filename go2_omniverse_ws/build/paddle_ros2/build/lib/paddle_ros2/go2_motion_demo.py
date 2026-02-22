#!/usr/bin/env python3
"""
go2_motion_demo.py  ·  Hardcoded Unitree Go2 motion sequence

Sends Twist messages to /robot0/cmd_vel:
1. Forward
2. Backward
3. Strafe Left
4. Strafe Right
5. Twist Left
6. Twist Right

Each motion lasts 2 sec, with 1 sec stop in between.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

# ───────── Twist helper ───────────────────────────────────────────
def make_twist(x=0.0, y=0.0, z=0.0):
    t = Twist()
    t.linear.x = x
    t.linear.y = y
    t.angular.z = z
    return t

# ───────── ROS 2 demo node ────────────────────────────────────────
class Go2MotionDemo(Node):
    def __init__(self):
        super().__init__("go2_motion_demo")
        self.pub = self.create_publisher(Twist, "/robot0/cmd_vel", 10)
        time.sleep(1.0)  # wait for pub connection

        self.get_logger().info("Starting hardcoded Go2 motion demo...")
        self.run_demo()

    def run_demo(self):
        sequence = [
            ("Move forward",  make_twist(x=1.0)),
            ("Move backward", make_twist(x=-1.0)),
            ("Strafe left",   make_twist(y=1.0)),
            ("Strafe right",  make_twist(y=-1.0)),
            ("Twist left",    make_twist(z=1.0)),
            ("Twist right",   make_twist(z=-1.0)),
        ]

        for label, cmd in sequence:
            self.get_logger().info(f"[DEMO] {label}")
            self.pub.publish(cmd)
            time.sleep(2.0)

            self.pub.publish(make_twist())  # stop
            time.sleep(1.0)

        self.get_logger().info("✅ Motion demo complete.")

# ───────── Entry point ────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    demo = Go2MotionDemo()
    rclpy.spin_once(demo, timeout_sec=0)  # just run once
    demo.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
