#!/usr/bin/env python3
"""
align_and_detect.py - 方向调整与目标检测工具
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py import point_cloud2
import numpy as np
import time
import os
import cv2
import sys
import yaml

# 添加PaddleDetection路径
sys.path.append('PaddleDetection')
from deploy.python.preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride
from deploy.python.utils import argsparser, Timer, get_current_memory_mb
from paddle.inference import Config
from paddle.inference import create_predictor

# ───────── 目标检测配置 ────────────────────────────────────────
class PredictConfig():
    def __init__(self, model_dir):
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf.get('min_subgraph_size', 3)
        self.labels = yml_conf['label_list']
        self.mask = yml_conf.get('mask', False)
        self.use_dynamic_shape = yml_conf.get('use_dynamic_shape', False)
        self.tracker = yml_conf.get('tracker', None)
        self.nms = yml_conf.get('NMS', None)
        self.fpn_stride = yml_conf.get('fpn_stride', None)

def load_predictor(model_dir):
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams')
    )
    config.enable_use_gpu(2000, 0)
    config.switch_ir_optim(False)
    config.disable_glog_info()
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)
    return predictor, config

def create_inputs(imgs, im_info):
    inputs = {}
    im_shape = []
    scale_factor = []
    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))
    origin_scale_factor = np.concatenate(scale_factor, axis=0)
    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    padding_imgs_shape = []
    padding_imgs_scale = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros((im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = np.array(img, dtype=np.float32)
        padding_imgs.append(padding_im)
        padding_imgs_shape.append(np.array([max_shape_h, max_shape_w]).astype('float32'))
        rescale = [float(max_shape_h) / float(im_h), float(max_shape_w) / float(im_w)]
        padding_imgs_scale.append(np.array(rescale).astype('float32'))
    inputs['image'] = np.stack(padding_imgs, axis=0)
    inputs['im_shape'] = np.stack(padding_imgs_shape, axis=0)
    inputs['scale_factor'] = origin_scale_factor
    return inputs

class Detector(object):
    def __init__(self, pred_config, model_dir):
        self.pred_config = pred_config
        self.predictor, self.config = load_predictor(model_dir)
        self.preprocess_ops = self.get_ops()

    def get_ops(self):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))
        return preprocess_ops

    def predict(self, inputs):
        input_names = self.predictor.get_input_names()
        for name in input_names:
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(inputs[name])
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        num_outs = int(len(output_names) / 2)
        np_boxes = self.predictor.get_output_handle(output_names[0]).copy_to_cpu()
        np_boxes_num = self.predictor.get_output_handle(output_names[num_outs]).copy_to_cpu()
        return dict(boxes=np_boxes, boxes_num=np_boxes_num)

# ───────── 方向调整与目标检测类 ────────────────────────────────
class AlignAndDetect(Node):
    def __init__(self, context=None, enable_alignment=True):
        """
        初始化方向调整与目标检测器

        参数:
            context: ROS 2 上下文对象 (可选)
            enable_alignment: 是否启用方向调整功能，默认 True
        """
        super().__init__("align_and_detect", context=context)
        self.enable_alignment = bool(enable_alignment)
        
        # 运动控制参数
        self.cmd_pub = self.create_publisher(Twist, "/robot0/cmd_vel", 10)
        
        # 订阅点云和里程计
        self.point_cloud_sub = self.create_subscription(
            PointCloud2, "/robot0/point_cloud2", self.point_cloud_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/robot0/odom", self.odom_callback, 10)
        
        # 订阅图像（用于目标检测）
        self.image_sub = self.create_subscription(
            Image, '/robot0/front_cam/rgb', self.image_callback, 10)
        
        # 机器人状态
        self.robot_position = Point()
        self.robot_yaw = 0.0
        
        # 参数设置
        self.center_deadzone = 0.1     # 中心点死区阈值（米）
        self.turn_speed = 0.5          # 转向速度（弧度/秒）
        self.align_count_threshold = 3  # 连续居中次数阈值
        self.fov_width = 0.5           # 视野宽度（米）
        self.min_points = 5             # 有效障碍物所需的最小点数
        self.ground_threshold = 0.1     # 地面高度阈值（米）
        
        # 目标检测参数
        self.det_model_path = "model/"
        self.threshold = 0.1            # 检测置信度阈值
        self.result_dir = "./result"    # 结果保存目录
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 状态变量
        self.state = "ALIGNING"         # 状态: ALIGNING, DETECTING, DONE
        self.obstacle_center_y = 0.0    # 障碍物在机器人坐标系中的横向位置
        self.align_count = 0            # 连续居中计数器
        self.detection_completed = False
        self.latest_image = None
        self.detector = None
        
        # 新增: 连续丢失障碍物相关参数
        self.consecutive_misses = 0      # 连续未检测到障碍物的次数
        self.consecutive_miss_threshold = 10  # 连续丢失阈值
        self.fov_enlarge_factor = 2.0    # FOV增大倍数
        self.original_fov_width = self.fov_width  # 原始视野宽度
        self.fov_enlarged = False        # 当前是否已增大FOV
        
        # 初始化目标检测器
        self.init_detector()
        
        # self.get_logger().info("方向调整与目标检测器已初始化")
    
    def init_detector(self):
        """初始化目标检测器"""
        try:
            self.pred_config = PredictConfig(self.det_model_path)
            self.detector = Detector(self.pred_config, self.det_model_path)
            # self.get_logger().info(f"目标检测模型加载成功。类别列表: {self.pred_config.labels}")
        except Exception as e:
            self.get_logger().error(f"加载目标检测模型失败: {e}")
            raise e
    
    def odom_callback(self, msg):
        """更新机器人位置和方向"""
        self.robot_position = msg.pose.pose.position
        
        # 从四元数计算偏航角（yaw）
        orientation = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y**2 + orientation.z**2)
        self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    def point_cloud_callback(self, msg):
        """处理点云数据，检测前方障碍物并计算横向位置"""
        if self.state != "ALIGNING":
            return
            
        try:
            # 提取点云数据
            points = point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)
            
            # 收集视野内的点（前方且横向在视野宽度内）
            fov_points = []
            
            for p in points:
                # 将点转换到机器人坐标系
                dx = p[0] - self.robot_position.x
                dy = p[1] - self.robot_position.y
                
                robot_frame_x = dx * np.cos(self.robot_yaw) + dy * np.sin(self.robot_yaw)
                robot_frame_y = -dx * np.sin(self.robot_yaw) + dy * np.cos(self.robot_yaw)
                
                # 只考虑机器人前方、视野宽度内且高于地面的点
                if (robot_frame_x > 0 and 
                    abs(robot_frame_y) <= self.fov_width/2 and
                    p[2] > self.ground_threshold):
                    fov_points.append(robot_frame_y)  # 只关心横向位置
            
            # 如果有足够的点，计算平均横向位置
            if len(fov_points) >= self.min_points:
                avg_y = sum(fov_points) / len(fov_points)
                self.obstacle_center_y = avg_y
                self.consecutive_misses = 0  # 重置连续丢失计数
                # self.get_logger().info(f"检测到障碍物横向偏移: {avg_y:.2f}米")
            else:
                self.obstacle_center_y = None
                self.consecutive_misses += 1
                # self.get_logger().info(f"视野内未检测到有效障碍物 (连续丢失: {self.consecutive_misses}/{self.consecutive_miss_threshold})")
                
            # 检查连续丢失次数
            if self.consecutive_misses >= self.consecutive_miss_threshold:
                if not self.fov_enlarged:
                    # 增大视野范围
                    self.original_fov_width = self.fov_width
                    self.fov_width *= self.fov_enlarge_factor
                    self.fov_enlarged = True
                    self.get_logger().warn(
                        f"连续{self.consecutive_miss_threshold}次未检测到目标，增大FOV至{self.fov_width:.2f}米")
                else:
                    # 已经增大视野后仍然找不到目标
                    self.get_logger().error(
                        f"增大FOV后仍然连续{self.consecutive_miss_threshold}次未检测到目标，停止目标检测")
                    # 停止机器人
                    self.cmd_pub.publish(Twist())
                    # 设置状态为完成
                    self.state = "DONE"
                    self.detection_completed = True
                
        except Exception as e:
            self.get_logger().error(f"处理点云数据时出错: {str(e)}")
            self.obstacle_center_y = None
            self.consecutive_misses += 1
    
    def image_callback(self, msg):
        """存储最新图像用于目标检测"""
        if self.state == "DETECTING":
            self.latest_image = msg
    
    def control_loop(self):
        """主控制循环（根据 enable_alignment 决定是否执行 ALIGNING）"""
        if self.state == "ALIGNING":
            if self.enable_alignment:
                self.align_obstacle()
            else:
                # 不启用方向调整，直接跳转到检测阶段
                # self.get_logger().info("方向调整已禁用，直接开始目标检测")
                self.state = "DETECTING"
        elif self.state == "DETECTING":
            self.detect_target()
    
    def align_obstacle(self):
        """调整机器人方向使障碍物居中"""
        cmd = Twist()
        
        if self.obstacle_center_y is None:
            # 未检测到障碍物，停止
            self.cmd_pub.publish(cmd)
            self.get_logger().info("未检测到障碍物，停止旋转")
            self.align_count = 0
            return
        
        # 检查障碍物是否已居中
        if abs(self.obstacle_center_y) < self.center_deadzone:
            self.align_count += 1
            # self.get_logger().info(f"障碍物已居中 ({self.align_count}/{self.align_count_threshold})")
            
            # 连续居中次数达到阈值
            if self.align_count >= self.align_count_threshold:
                # self.get_logger().info("方向调整完成，开始目标检测")
                self.cmd_pub.publish(cmd)  # 停止
                # 如果扩大了视野，恢复原始视野
                if self.fov_enlarged:
                    self.fov_width = self.original_fov_width
                    self.fov_enlarged = False
                    self.get_logger().info(f"恢复FOV宽度至{self.fov_width:.2f}米")
                self.state = "DETECTING"
                return
        else:
            # 重置计数器
            self.align_count = 0
            
            # 计算旋转方向
            turn_direction = 1 if self.obstacle_center_y < 0 else -1
            cmd.angular.z = -turn_direction * self.turn_speed
            
            # 发布旋转命令
            self.cmd_pub.publish(cmd)
            # self.get_logger().info(f"旋转调整中: 偏移={self.obstacle_center_y:.2f}米, 速度={cmd.angular.z:.2f} rad/s")
    
    def detect_target(self):
        """执行目标检测并保存结果"""
        if self.latest_image is None:
            self.get_logger().info("等待图像数据...")
            return
            
        try:
            # 转换图像数据
            dtype = np.uint8
            image_np = np.frombuffer(self.latest_image.data, dtype=dtype)
            
            # 处理图像格式
            if self.latest_image.encoding == 'rgb8' or self.latest_image.encoding == 'bgr8':
                image_np = image_np.reshape((self.latest_image.height, self.latest_image.width, 3))
                if self.latest_image.encoding == 'rgb8':
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                self.get_logger().warn(f"不支持的图像格式: {self.latest_image.encoding}")
                self.state = "DONE"
                return
            
            # 处理图像并检测目标
            self.process_and_save_detection(image_np)
            
            # 标记完成
            self.state = "DONE"
            self.detection_completed = True
            
        except Exception as e:
            self.get_logger().error(f"目标检测失败: {e}")
            self.state = "DONE"
            self.detection_completed = True
    
    def process_and_save_detection(self, image_np):
        """处理图像进行目标检测并保存结果，只输出置信度最大的目标"""
        try:
            # 预处理图像
            input_im_lst = []
            input_im_info_lst = []
            
            # 使用临时文件处理图像
            temp_file = "/tmp/go2_temp_img.jpg"
            cv2.imwrite(temp_file, image_np)
            
            im, im_info = preprocess(temp_file, self.detector.preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
            inputs = create_inputs(input_im_lst, input_im_info_lst)
            
            # 进行预测
            det_results = self.detector.predict(inputs)
            
            # 处理预测结果
            im_bboxes_num = det_results['boxes_num'][0]
            detected_objects = []  # 存储所有检测到的物体
            
            if im_bboxes_num > 0:
                bbox_results = det_results['boxes'][0:im_bboxes_num, 2:]
                id_results = det_results['boxes'][0:im_bboxes_num, 0]
                score_results = det_results['boxes'][0:im_bboxes_num, 1]
                
                for idx in range(im_bboxes_num):
                    class_id = int(id_results[idx])
                    confidence = float(score_results[idx])
                    
                    # 检查置信度是否足够高
                    if confidence >= self.threshold:
                        x1, y1, x2, y2 = bbox_results[idx]
                        detected_objects.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': self.pred_config.labels[class_id]
                        })
            
            # 只保留置信度最大的目标
            max_confidence_object = None
            if detected_objects:
                # 按置信度降序排序并取第一个
                detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
                max_confidence_object = detected_objects[0]
            
            # 输出检测结果
            if max_confidence_object:
                obj = max_confidence_object
                self.get_logger().info(
                    f"✅ 检测到目标: {obj['class_name']}, "
                    f"置信度: {obj['confidence']:.2f}, "
                    f"位置: [{obj['bbox'][0]:.0f}, {obj['bbox'][1]:.0f}, "
                    f"{obj['bbox'][2]:.0f}, {obj['bbox'][3]:.0f}]"
                )
            else:
                self.get_logger().info("⚠️ 未检测到目标")
            
            # 保存带检测框的图像（只绘制最大置信度目标）
            self.save_annotated_image(image_np, [max_confidence_object] if max_confidence_object else [])
            
        except Exception as e:
            self.get_logger().error(f"目标检测处理失败: {e}")
    
    def save_annotated_image(self, image_np, detected_objects):
        """保存带检测框和标签的图像"""
        # 复制原始图像
        annotated_img = image_np.copy()
        
        # 为每个检测到的物体绘制边界框和标签
        for obj in detected_objects:
            # 绘制边界框
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), 
                          (0, 255, 0), 2)
            
            # 添加标签和置信度
            label = f"{obj['class_name']}: {obj['confidence']:.2f}"
            cv2.putText(annotated_img, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存图像
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(self.result_dir, f"detection_{timestamp}.jpg")
        cv2.imwrite(img_path, annotated_img)
        self.get_logger().info(f"📸 检测结果已保存至: {img_path}")
    
    def run(self):
        """运行方向调整与目标检测"""
        # self.get_logger().info("开始方向调整与目标检测...")
        self.detection_completed = False
        
        # 创建控制定时器
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # 等待过程完成
        while rclpy.ok() and not self.detection_completed:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # 完成后停止机器人
        self.cmd_pub.publish(Twist())
        self.get_logger().info("方向调整与目标检测完成")
        
        # 清理资源
        self.destroy_node()

# ───────── 外部调用接口 ───────────────────────────────────────
def align_and_detect(context=None, enable_alignment=True):
    """
    执行方向调整与目标检测

    参数:
        context: ROS 2 上下文对象 (可选)
        enable_alignment: 是否启用方向调整功能，默认 True
    """
    node = AlignAndDetect(context=context, enable_alignment=enable_alignment)
    node.run()