import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from paddleocr import PaddleOCR
import cv2
import time

class OCRNode(Node):
    def __init__(self):
        super().__init__('ocr_node')

        self.subscription = self.create_subscription(
            Image,
            '/robot0/front_cam/rgb',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(String, '/ocr_result', 10)
        self.bridge = CvBridge()
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.last_ocr_time = 0
        self.ocr_interval = 0.7  # seconds

        self.latest_image = None
        self.create_timer(0.1, self.run_ocr)

    def image_callback(self, msg):
        try:
            # Save the latest image (overwrite previous)
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def run_ocr(self):
        now = time.time()
        if self.latest_image is not None and now - self.last_ocr_time >= self.ocr_interval:
            result = self.ocr.ocr(self.latest_image, cls=True)
            lines = [line[1][0] for line in result[0]]
            output_text = '\n'.join(lines)

            msg = String()
            msg.data = output_text
            self.publisher.publish(msg)

            self.get_logger().info(f"OCR Output:\n{output_text}")
            self.last_ocr_time = now

def main(args=None):
    rclpy.init(args=args)
    node = OCRNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
