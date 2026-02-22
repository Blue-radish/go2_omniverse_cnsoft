#!/home/ziyueg/miniconda3/envs/paddle_env/bin/python
"""
ROS 2 node that takes /robot0/front_cam/rgb (sensor_msgs/Image),
runs PaddleOCR.predict(), and publishes recognised text
(one string per line) to /ocr_result.
"""

import os, time, json, cv2, rclpy
from rclpy.node      import Node
from sensor_msgs.msg import Image
from std_msgs.msg    import String
from cv_bridge       import CvBridge
from paddleocr       import PaddleOCR


class OCRNode(Node):
    def __init__(self):
        super().__init__("ocr_node")

        # ───────── parameters (override in launch if you like)
        self.declare_parameter("topic_in",   "/robot0/front_cam/rgb")
        self.declare_parameter("topic_out",  "/ocr_result")
        self.declare_parameter("lang",       "ch")        # Chinese
        self.declare_parameter("period",     1.5)         # seconds
        self.declare_parameter("debug_dir",  "/tmp/ocr_dbg")

        topic_in   = self.get_parameter("topic_in").value
        topic_out  = self.get_parameter("topic_out").value
        lang       = self.get_parameter("lang").value
        self.period     = self.get_parameter("period").value
        self.debug_dir  = self.get_parameter("debug_dir").value
        os.makedirs(self.debug_dir, exist_ok=True)

        # ───────── comms
        self.sub = self.create_subscription(Image, topic_in, self._cb_img, 10)
        self.pub = self.create_publisher  (String, topic_out, 10)
        self.bridge = CvBridge()

        # ───────── OCR engine
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=lang,
        )

        self.buffer     = None
        self.last_stamp = 0.0
        self.timer = self.create_timer(0.05, self._tick)

        self.get_logger().info(
            f"OCR node ready  in:{topic_in}  out:{topic_out}  lang:{lang}"
        )

    # ───────────────────────────────────────────────────────── callbacks
    def _cb_img(self, msg: Image):
        try:
            self.buffer = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge: {exc}")

    def _tick(self):
        if self.buffer is None:
            return
        now = time.time()
        if now - self.last_stamp < self.period:
            return

        img = self.buffer.copy()
        cv2.imwrite(f"{self.debug_dir}/input_{int(now)}.jpg", img)

        try:
            # ---- low-level predict() call (as you had before)
            results = self.ocr.predict(img)
            lines = []

            if results and len(results) > 0:
                ocr_obj = results[0]      # list with one OCRResult/dict
                if isinstance(ocr_obj, dict) and "rec_texts" in ocr_obj:
                    lines = ocr_obj["rec_texts"]
                elif hasattr(ocr_obj, "rec_texts"):
                    lines = ocr_obj.rec_texts
                else:
                    # dump unknown structure once for debugging
                    dumpable = (
                        {k: str(v) for k, v in ocr_obj.items()}
                        if isinstance(ocr_obj, dict)
                        else {k: str(v) for k, v in ocr_obj.__dict__.items()}
                    )
                    self.get_logger().warn(
                        "Unrecognised OCRResult format:\n" +
                        json.dumps(dumpable, indent=2, ensure_ascii=False)
                    )

            txt = "\n".join(lines)
            self.pub.publish(String(data=txt))
            self.get_logger().info(f"OCR:\n{txt or '(none)'}")

        except Exception as exc:
            self.get_logger().error(f"PaddleOCR.predict failed: {exc}")

        self.last_stamp = now


def main(args=None):
    rclpy.init(args=args)
    node = OCRNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
