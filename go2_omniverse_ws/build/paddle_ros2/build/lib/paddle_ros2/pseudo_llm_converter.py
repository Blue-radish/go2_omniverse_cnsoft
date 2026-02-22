#!/home/ziyueg/miniconda3/envs/paddle_env/bin/python
"""
pseudo_llm_converter.py
=======================

Subscribe to /ocr_result (String) coming from paddle_ros2.ocr_node,
translate keywords into Twist commands, publish on /robot0/cmd_vel.

If no matching OCR text is received for ``timeout`` seconds, it
automatically republishes a *zero* Twist so the robot stops.

Usage
-----
# after sourcing ROS 2 Humble, in the paddle_env conda env
ros2 run paddle_ros2 pseudo_llm_converter
# …or give it a unique node name
ros2 run paddle_ros2 pseudo_llm_converter \
       --ros-args -r __node:=pseudo_llm_converter2
"""

import re
import rclpy
from rclpy.node        import Node
from std_msgs.msg      import String
from geometry_msgs.msg import Twist


class ConverterNode(Node):
    """
    Map recognised text → Twist with a small state-machine timeout.
    """

    def __init__(self, name: str = "pseudo_llm_converter") -> None:
        super().__init__(name)

        # ROS I/O -------------------------------------------------------------
        self.sub = self.create_subscription(String,
                                            "/ocr_result",
                                            self._cb_ocr,
                                            10)
        self.pub = self.create_publisher(Twist,
                                         "/robot0/cmd_vel",
                                         10)

        # Regex → (Twist field, value)
        self.patterns = {
            re.compile(r"左前方"):                 ("angular.z", +1.0),
            re.compile(r"右前方"):                 ("angular.z", -1.0),
            re.compile(r"(go\s*forward|前进)"):    ("linear.x",  +1.0),
            re.compile(r"(back|)"):            ("linear.x",  -1.0),
        }

        # Timeout parameters --------------------------------------------------
        self.cmd_cache = Twist()          # last non-zero command
        self.last_time = 0.0              # when it was received (sec)
        self.timeout   = 0.50             # keep it alive this long (sec)

        # Clock-driven publisher
        self.timer = self.create_timer(0.05, self._tick)   # 20 Hz

        self.get_logger().info("pseudo_llm_converter ready")

    # ───────────────────────────── callbacks ──────────────────────────────
    def _cb_ocr(self, msg: String) -> None:
        """Handle incoming OCR result text."""
        txt = msg.data.strip().lower()
        self.get_logger().debug(f"OCR text: «{txt}»")

        cmd = Twist()
        matched = False

        for pat, (field, val) in self.patterns.items():
            if pat.search(txt):
                ns, attr = field.split(".")
                getattr(cmd, ns).__setattr__(attr, val)
                matched = True
                break                        # stop after first match

        if matched:
            self.cmd_cache = cmd
            self.last_time = self.get_clock().now().seconds_nanoseconds()[0]
            self.pub.publish(cmd)
            self.get_logger().info(f"→ cmd_vel {cmd}")
        else:
            self.get_logger().debug("No keyword matched")

    def _tick(self) -> None:
        """Republish cached Twist or send zero when it expires."""
        now = self.get_clock().now().seconds_nanoseconds()[0]
        if now - self.last_time < self.timeout:
            # still fresh – keep sending the cached command
            if self.cmd_cache.linear.x or self.cmd_cache.angular.z:
                self.pub.publish(self.cmd_cache)
        else:
            # timeout reached – send stop once, then stay silent
            if self.cmd_cache.linear.x or self.cmd_cache.angular.z:
                self.get_logger().info("⌛ timeout – sending stop")
                self.cmd_cache = Twist()
                self.pub.publish(self.cmd_cache)

# ───────────────────────────── main entry ───────────────────────────────
def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConverterNode()        # default name or override via ROS remap
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
