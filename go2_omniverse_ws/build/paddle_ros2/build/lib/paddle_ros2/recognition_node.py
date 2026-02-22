# 此节点为伪节点，尚未实现图像识别功能并接入PADDLE_ROS2
# 具体功能请自行实现

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import time
import paddlehub as hub

class RecognitionNode(Node):
    def __init__(self):
        super().__init__('recognition_node')

        self.subscription = self.create_subscription(
            Image,
            '/robot0/front_cam/rgb',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(String, '/recognition_result', 10)
        self.bridge = CvBridge()

        # 使用自定义训练的 PaddleRecognition 模型
        # 假设自定义模型已保存在 './custom_model' 路径下
        self.module = hub.Module(name='/home/hunch/go2_omniverse_cnsoft')

        self.last_recognition_time = 0
        self.recognition_interval = 1.5  # seconds
        self.latest_image = None
        self.create_timer(0.1, self.run_recognition)

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def run_recognition(self):
        now = time.time()
        if self.latest_image is not None and now - self.last_recognition_time >= self.recognition_interval:
            try:
                # 执行图像识别
                result = self.module.predict(self.latest_image)

                # 假设识别结果为一个包含类别和置信度的列表
                # 根据具体模型输出格式调整解析方式
                output_text = ''
                for item in result:
                    output_text += f"Class: {item['class']}, Confidence: {item['confidence']:.4f}\n"

                msg = String()
                msg.data = output_text
                self.publisher.publish(msg)

                self.get_logger().info(f"[Recognition Result]\n{output_text}")
                self.last_recognition_time = now
            except Exception as e:
                self.get_logger().error(f"Recognition failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RecognitionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()