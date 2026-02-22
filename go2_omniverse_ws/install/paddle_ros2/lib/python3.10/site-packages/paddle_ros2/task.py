#!/usr/bin/env python3
"""
task.py · Modular task control with sequential state machine
运动函数基于输入方向和持续时间执行
状态序列易于扩展新步骤
"""

import re
import time
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from paddleocr import PaddleOCR
from openai import OpenAI
from dotenv import load_dotenv

# =====================================================================
# 常量配置
# =====================================================================
DEFAULT_VELOCITY = 2.0  # 默认运动速度

# ERNIE API 配置
# load_dotenv()
# ERNIE_CLIENT = OpenAI(
#     api_key="34aaecf280b39a57b980e1f35a881756a0c19572",
#     base_url="https://aistudio.baidu.com/llm/lmapi/v3",
# )

# # OCR 配置
# OCR = PaddleOCR(
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
#     lang="ch",  # 中文识别
# )

# SYSTEM_PROMPT = (
#     "你是指令解释器。根据文本返回单个动作词："
#     "left (左转), right (右转), forward (前进), back (后退)"
# )

# =====================================================================
# 运动控制函数
# =====================================================================
def create_twist(x=0.0, y=0.0, z=0.0):
    """创建 Twist 消息"""
    t = Twist()
    t.linear.x = x
    t.linear.y = y
    t.angular.z = z
    return t

def execute_movement(cmd_pub, direction, duration, velocity=DEFAULT_VELOCITY):
    """
    执行指定方向的运动
    
    参数:
        cmd_pub: ROS 发布器
        direction: 运动方向 ('forward', 'back', 'left', 'right')
        duration: 运动持续时间 (秒)
        velocity: 运动速度 (可选)
    """
    start_time = time.time()
    
    # 映射方向到运动命令
    if direction == "forward":
        cmd = create_twist(x=velocity)
    elif direction == "back":
        cmd = create_twist(x=-velocity)
    elif direction == "left":
        cmd = create_twist(z=velocity)  # 正值为左转
    elif direction == "right":
        cmd = create_twist(z=-velocity)  # 负值为右转
    else:  # 未知方向默认为前进
        cmd = create_twist(x=velocity)
    
    # 执行运动
    while (time.time() - start_time) < duration:
        cmd_pub.publish(cmd)
        time.sleep(0.05)  # 20Hz控制频率
    
    # 停止
    cmd_pub.publish(create_twist())
    time.sleep(5.0)

# =====================================================================
# OCR 与 AI 处理
# =====================================================================
def capture_and_recognize(bridge, camera_image, logger):
    """捕获图像并执行OCR识别"""
    if camera_image is None:
        logger.warn("无图像可用于OCR")
        return ""
    
    try:
        # 执行OCR
        results = OCR.predict(camera_image)
        
        # 处理OCR结果
        if results and isinstance(results[0], dict) and "rec_texts" in results[0]:
            return " ".join(results[0]["rec_texts"])
        return "OCR未识别到文本"
    except Exception as e:
        logger.error(f"OCR处理失败: {e}")
        return ""

# def interpret_direction(text, logger):
#     """解释文本中的方向指令"""
#     text = text.lower()
    
#     # 规则匹配
#     direction_rules = {
#         "left": ["左", "turn left", "left"],
#         "right": ["右", "turn right", "right"],
#         "forward": ["前", "go forward", "forward"],
#         "back": ["后", "go back", "back"]
#     }
    
#     # for direction, keywords in direction_rules.items():
#     #     if any(kw in text for kw in keywords):
#     #         logger.info(f"规则匹配: {text} → {direction}")
#     #         return direction
    
#     # 使用ERNIE进行理解
#     try:
#         logger.info(f"使用ERNIE理解指令: {text}")
#         response = ERNIE_CLIENT.chat.completions.create(
#             model="ernie-3.5-8k",
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": text},
#             ],
#             temperature=0.0,
#             max_tokens=10
#         )
#         answer = response.choices[0].message.content.lower().strip()
#         logger.info(f"ERNIE回复: {answer}")
        
#         # 解析ERNIE的回复
#         if any(kw in answer for kw in direction_rules["left"]):
#             return "left"
#         elif any(kw in answer for kw in direction_rules["right"]):
#             return "right"
#         elif any(kw in answer for kw in direction_rules["forward"]):
#             return "forward"
#         elif any(kw in answer for kw in direction_rules["back"]):
#             return "back"
#     except Exception as e:
#         logger.error(f"ERNIE请求失败: {e}")
    
#     return "forward"  # 默认方向

# =====================================================================
# 任务控制节点
# =====================================================================
class TaskController(Node):
    def __init__(self):
        super().__init__("task_controller")
        
        # ROS 通信
        self.cmd_pub = self.create_publisher(Twist, "/robot0/cmd_vel", 10)
        self.camera_sub = self.create_subscription(Image, "/robot0/front_cam/rgb", self.camera_callback, 10)
        
        # 工具初始化
        self.bridge = CvBridge()
        self.camera_image = None
        self.ocr_lock = threading.Lock()
        
        # 任务状态
        self.task_thread = None
        self.running = False
        
        self.get_logger().info("任务控制器准备就绪 - 输入 '/start' 启动任务")
    
    def camera_callback(self, msg):
        """存储最新图像"""
        with self.ocr_lock:
            try:
                self.camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                self.get_logger().error(f"图像转换错误: {e}")
    
    def start_task(self):
        """启动任务序列"""
        if self.running:
            self.get_logger().warn("任务已在运行中")
            return
        
        self.running = True
        self.task_thread = threading.Thread(target=self.task_sequence)
        self.task_thread.daemon = True
        self.task_thread.start()
        self.get_logger().info("任务启动!")
    
    def task_sequence(self):
        """定义任务执行序列 - 可轻松扩展"""
        logger = self.get_logger()
        
        # 第一阶段：向前移动
        execute_movement(self.cmd_pub, "back", 3.0)
        logger.info("阶段1: 向前移动5秒")
        execute_movement(self.cmd_pub, "forward", 8.0)
        logger.info("阶段1: 向前移动5秒")
        execute_movement(self.cmd_pub, "left", 5.0)
        logger.info("阶段1: 向前移动5秒")
        execute_movement(self.cmd_pub, "forward", 5.0)
        logger.info("阶段1: 向前移动5秒")
        execute_movement(self.cmd_pub, "right", 5.0)
        execute_movement(self.cmd_pub, "forward", 8.0)
        execute_movement(self.cmd_pub, "left",5.0)
        execute_movement(self.cmd_pub, "forward", 7.0)
        execute_movement(self.cmd_pub, "left", 5.0)
        logger.info("阶段1: 向前移动5秒")
        execute_movement(self.cmd_pub, "forward", 10.0)
        logger.info("阶段1: 向前移动5秒")
        execute_movement(self.cmd_pub, "right", 5.0)
        logger.info("阶段1: 向前移动5秒")
        execute_movement(self.cmd_pub, "forward", 5.0)
        execute_movement(self.cmd_pub, "right", 5.0)
        execute_movement(self.cmd_pub, "forward", 10.0)


    #     def task_sequence(self):
    #     """定义任务执行序列 - 可轻松扩展"""
    #     logger = self.get_logger()
        
    #     try:
    #         # 第一阶段：向前移动
    #         logger.info("阶段1: 向前移动5秒")
    #         execute_movement(self.cmd_pub, "forward", 5.0)
            
    #         # 第二阶段：捕获图像并识别
    #         logger.info("阶段2: 图像捕获与识别")
    #         time.sleep(1.0)  # 等待稳定
    #         with self.ocr_lock:
    #             ocr_text = capture_and_recognize(self.bridge, self.camera_image, logger)
    #         logger.info(f"OCR结果: {ocr_text}")
            
    #         # 第三阶段：根据指令转向
    #         direction = interpret_direction(ocr_text, logger)
    #         logger.info(f"解析方向: {direction}")
    #         logger.info(f"阶段3: 向{direction}移动2秒")
    #         execute_movement(self.cmd_pub, direction, 2.0)
            
    #         # 第四阶段：再次向前移动
    #         logger.info("阶段4: 向前移动5秒")
    #         execute_movement(self.cmd_pub, "forward", 5.0)
            
    #         # 任务完成
    #         logger.info("✅ 任务完成!")
            
    #     except Exception as e:
    #         logger.error(f"任务执行出错: {e}")
    #     finally:
    #         self.running = False
    

# =====================================================================
# 主执行入口
# =====================================================================
def main(args=None):
    rclpy.init(args=args)
    controller = TaskController()
    
    # 简单命令行界面 (在实际应用中可替换为ROS服务)
    def user_input():
        while rclpy.ok():
            cmd = input("输入命令 (输入 '1' 运行任务): ")
            if cmd == "1":
                controller.start_task()
            elif cmd == "2":
                break
            time.sleep(0.1)
    
    # 启动命令行界面线程
    input_thread = threading.Thread(target=user_input)
    input_thread.daemon = True
    input_thread.start()
    
    # 运行ROS节点
    rclpy.spin(controller)
    
    # 清理
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
