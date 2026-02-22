#!/usr/bin/env python3
"""
visual_tracker.py - 视觉追踪辅助工具

功能:
1. 订阅相机图像并进行目标检测
2. 调整机器人位置使目标中心保持在视野中心
3. 当目标连续三帧未检测到时停止行走
4. 在终端输出目标类别和置信度
5. 保存绘制有框框的图片到./result目录
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import numpy as np
import cv2
import os
import time
import sys
import yaml
from collections import deque

# 添加PaddleDetection路径
sys.path.append('PaddleDetection')
from deploy.python.preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride
from paddle.inference import Config
from paddle.inference import create_predictor

# ───────── 目标检测相关类 ───────────────────────────────────────────
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

# ───────── 视觉追踪工具类 ─────────────────────────────────────────
class VisualTracker(Node):
    def __init__(self, context=None):
        """
        初始化视觉追踪器
        
        参数:
            context: ROS 2 上下文对象 (可选)
        """
        super().__init__("visual_tracker", context=context)
        
        # 运动控制参数
        self.cmd_pub = self.create_publisher(Twist, "/robot0/cmd_vel", 10)
        self.move_forward_speed = 0.4  # 前进速度
        self.rotation_gain = 0.005     # 旋转增益系数
        self.stop_threshold = 0.05     # 停止旋转的偏移阈值
        self.max_rotation = 1.0        # 最大旋转速度
        
        # 目标检测参数
        self.det_model_path = "model/"
        self.threshold = 0.3           # 检测置信度阈值
        self.target_class = "ball"     # 目标类别名称
        self.target_class_id = None     # 目标类别ID
        
        # 图像保存参数
        self.result_dir = "./result"
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 初始化检测器
        try:
            self.pred_config = PredictConfig(self.det_model_path)
            self.detector = Detector(self.pred_config, self.det_model_path)
            
            # 查找目标类别的ID
            if self.target_class_id is None:
                for i, label in enumerate(self.pred_config.labels):
                    if self.target_class.lower() in label.lower():
                        self.target_class_id = i
                        break
                if self.target_class_id is None:
                    self.target_class_id = 0  # 默认使用第一个类别
                    self.get_logger().warn(f"目标类别 '{self.target_class}' 未找到，使用第一个类别")
            
            self.get_logger().info(f"模型加载成功。目标类别: {self.pred_config.labels[self.target_class_id]}")
        except Exception as e:
            self.get_logger().error(f"加载模型失败: {e}")
            raise e
        
        # 图像订阅
        self.image_sub = self.create_subscription(
            Image,
            '/robot0/front_cam/rgb',
            self.image_callback,
            10
        )
        
        # 状态变量
        self.last_image_time = 0
        self.processing_interval = 0.3  # 处理间隔(秒)
        self.detection_history = deque(maxlen=3)  # 存储最近3帧的检测结果
        self.active = True
        self.best_target = None  # 存储最佳目标信息
        
        self.get_logger().info("视觉追踪器已就绪，等待图像...")

    def image_callback(self, msg):
        # 检查处理间隔
        current_time = time.time()
        if current_time - self.last_image_time < self.processing_interval:
            return
        self.last_image_time = current_time
        
        try:
            # 转换图像数据
            dtype = np.uint8
            image_np = np.frombuffer(msg.data, dtype=dtype)
            
            # 处理图像格式
            if msg.encoding == 'rgb8' or msg.encoding == 'bgr8':
                image_np = image_np.reshape((msg.height, msg.width, 3))
                if msg.encoding == 'rgb8':
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                self.get_logger().warn(f"不支持的图像格式: {msg.encoding}")
                return
            
            # 处理图像并检测目标
            target_center = self.process_image(image_np)
            
            # 更新检测历史
            self.detection_history.append(target_center is not None)
            
            # 控制决策
            self.control_robot(target_center, msg.width, msg.height)
            
            # 保存带检测框的图像
            self.save_annotated_image(image_np)
            
        except Exception as e:
            self.get_logger().error(f"图像处理失败: {e}")

    def process_image(self, image_np):
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
            self.best_target = None  # 重置最佳目标
            max_confidence = 0.0
            
            if im_bboxes_num > 0:
                bbox_results = det_results['boxes'][0:im_bboxes_num, 2:]
                id_results = det_results['boxes'][0:im_bboxes_num, 0]
                score_results = det_results['boxes'][0:im_bboxes_num, 1]
                
                for idx in range(im_bboxes_num):
                    class_id = int(id_results[idx])
                    confidence = float(score_results[idx])
                    
                    # 检查是否为目标类别且置信度足够高
                    if class_id == self.target_class_id and confidence >= self.threshold:
                        # 更新最佳目标
                        if confidence > max_confidence:
                            max_confidence = confidence
                            x1, y1, x2, y2 = bbox_results[idx]
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            self.best_target = {
                                'center': (center_x, center_y),
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'class_id': class_id
                            }
            
            # 输出检测结果
            if self.best_target:
                class_name = self.pred_config.labels[self.best_target['class_id']]
                self.get_logger().info(f"检测到目标: {class_name}, 置信度: {self.best_target['confidence']:.2f}")
            
            return self.best_target['center'] if self.best_target else None
        
        except Exception as e:
            self.get_logger().error(f"推理失败: {e}")
            return None

    def save_annotated_image(self, image_np):
        """保存带检测框和标签的图像"""
        if not self.best_target:
            return
            
        # 复制原始图像
        annotated_img = image_np.copy()
        
        # 绘制边界框
        x1, y1, x2, y2 = self.best_target['bbox']
        cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), 
                      (0, 255, 0), 2)
        
        # 添加标签和置信度
        class_name = self.pred_config.labels[self.best_target['class_id']]
        label = f"{class_name}: {self.best_target['confidence']:.2f}"
        cv2.putText(annotated_img, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存图像
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(self.result_dir, f"detection_{timestamp}.jpg")
        cv2.imwrite(img_path, annotated_img)
        self.get_logger().info(f"保存检测结果到: {img_path}")

    def control_robot(self, target_center, image_width, image_height):
        # 检查是否需要停止
        if not self.active:
            return
            
        # 创建运动命令
        cmd = Twist()
        
        # 检查目标是否连续三帧未检测到
        if len(self.detection_history) == 3 and not any(self.detection_history):
            self.get_logger().info("连续三帧未检测到目标，停止运动")
            self.cmd_pub.publish(cmd)  # 发布停止命令
            self.active = False
            return
        
        # 如果没有检测到目标，停止运动
        if target_center is None:
            self.get_logger().info("未检测到目标，停止运动")
            self.cmd_pub.publish(cmd)
            return
        
        # 计算目标中心偏移
        target_x, target_y = target_center
        image_center_x = image_width / 2
        offset_x = target_x - image_center_x
        normalized_offset = offset_x / image_width
        
        # 计算旋转速度 (比例控制)
        rotation_speed = -self.rotation_gain * offset_x
        
        # 应用限幅
        if abs(rotation_speed) > self.max_rotation:
            rotation_speed = self.max_rotation if rotation_speed > 0 else -self.max_rotation
        
        # 设置运动命令
        cmd.angular.z = rotation_speed
        
        # 如果偏移足够小，前进
        if abs(normalized_offset) < self.stop_threshold:
            cmd.linear.x = self.move_forward_speed
            self.get_logger().info(f"目标已居中，以 {self.move_forward_speed:.2f} m/s 前进")
        else:
            self.get_logger().info(f"调整旋转: {rotation_speed:.3f} rad/s (偏移: {offset_x:.1f} px)")
        
        # 发布运动命令
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        """安全停止机器人"""
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        self.get_logger().info("机器人已停止")
        
    def track_until_lost(self):
        """开始追踪直到目标丢失"""
        self.active = True
        self.detection_history.clear()
        
        # 重置状态并开始追踪
        while rclpy.ok() and self.active:
            rclpy.spin_once(self, timeout_sec=0.1)
            
        # 返回是否因为目标丢失而停止
        return True