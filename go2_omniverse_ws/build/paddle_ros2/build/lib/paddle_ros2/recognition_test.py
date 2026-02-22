# 此节点实现拍照、推理和保存结果循环
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
import os
import time
import sys
import json
import yaml
from functools import reduce
import multiprocessing
from PIL import Image as PILImage
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

# 添加PaddleDetection路径
sys.path.append('PaddleDetection')
from deploy.python.preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride
from deploy.python.utils import argsparser, Timer, get_current_memory_mb

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
        self.print_config()

    def print_config(self):
        print('%s: %s' % ('Model Arch', self.arch))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))

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
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
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

class RecognitionNode(Node):
    def __init__(self):
        super().__init__('recognition_node')
        
        # 创建必要的目录
        self.image_dir = './temp'
        self.result_dir = './result'
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 初始化模型
        self.det_model_path = "model/"
        self.threshold = 0.3
        self.pred_config = PredictConfig(self.det_model_path)
        self.detector = Detector(self.pred_config, self.det_model_path)
        self.get_logger().info("Model loaded successfully")
        
        # 订阅相机话题
        self.subscription = self.create_subscription(
            Image,
            '/robot0/front_cam/rgb',
            self.image_callback,
            10
        )
        
        # 控制推理频率
        self.last_process_time = 0
        self.process_interval = 1.5  # 处理间隔(秒)
        self.image_counter = 0

    def image_callback(self, msg):
        # 检查是否达到处理间隔
        current_time = time.time()
        if current_time - self.last_process_time < self.process_interval:
            return
            
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
                self.get_logger().warn(f"Unsupported image encoding: {msg.encoding}")
                return
                
            # 保存原始图片到临时目录
            timestamp = int(time.time() * 1000)  # 毫秒级时间戳
            filename = f"temp_{timestamp}.jpg"
            filepath = os.path.join(self.image_dir, filename)
            cv2.imwrite(filepath, image_np)
            self.get_logger().info(f"Saved temp image: {filepath}")
            
            # 进行推理
            self.process_image(filepath)
            
            # 更新状态
            self.last_process_time = current_time
            self.image_counter += 1
            
        except Exception as e:
            self.get_logger().error(f"Image processing failed: {e}")

    def process_image(self, image_path):
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 预处理图像
            input_im_lst = []
            input_im_info_lst = []
            im, im_info = preprocess(image_path, self.detector.preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
            inputs = create_inputs(input_im_lst, input_im_info_lst)
            
            # 进行预测
            det_results = self.detector.predict(inputs)
            
            # 处理预测结果
            im_bboxes_num = det_results['boxes_num'][0]
            results = []
            
            if im_bboxes_num > 0:
                bbox_results = det_results['boxes'][0:im_bboxes_num, 2:]
                id_results = det_results['boxes'][0:im_bboxes_num, 0]
                score_results = det_results['boxes'][0:im_bboxes_num, 1]
                
                for idx in range(im_bboxes_num):
                    if float(score_results[idx]) >= self.threshold:
                        # 提取边界框坐标
                        x1, y1, x2, y2 = bbox_results[idx]
                        class_id = int(id_results[idx])
                        confidence = float(score_results[idx])
                        
                        # 添加到结果列表
                        results.append({
                            "class_id": class_id,
                            "confidence": confidence,
                            "bbox": [x1, y1, x2, y2]
                        })
            
            # 打印结果
            self.print_results(results)
            
            # 绘制并保存结果图像
            self.draw_and_save_results(image_path, results)
            
            # 记录处理时间
            process_time = time.time() - start_time
            self.get_logger().info(f"Inference time: {process_time:.3f}s")
            
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()

    def print_results(self, results):
        if not results:
            print("No objects detected")
            return
            
        print("\n===== Detection Results =====")
        for i, result in enumerate(results):
            class_id = result["class_id"]
            confidence = result["confidence"]
            bbox = result["bbox"]
            class_name = self.pred_config.labels[class_id] if class_id < len(self.pred_config.labels) else f"Class {class_id}"
            
            print(f"Object {i+1}:")
            print(f"  Class: {class_name} (ID: {class_id})")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Bounding Box: x1={bbox[0]:.1f}, y1={bbox[1]:.1f}, x2={bbox[2]:.1f}, y2={bbox[3]:.1f}")
            print(f"  Dimensions: width={bbox[2]-bbox[0]:.1f}, height={bbox[3]-bbox[1]:.1f}")
            print("")
        print("============================")

    def draw_and_save_results(self, image_path, results):
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            self.get_logger().error(f"Failed to read image: {image_path}")
            return
            
        # 绘制检测结果
        for result in results:
            class_id = result["class_id"]
            confidence = result["confidence"]
            x1, y1, x2, y2 = map(int, result["bbox"])
            
            # 获取类别名称
            class_name = self.pred_config.labels[class_id] if class_id < len(self.pred_config.labels) else f"Class {class_id}"
            
            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1), (x1 + label_width, y1 - label_height - 5), color, -1)
            
            # 绘制标签文本
            cv2.putText(image, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 保存结果图像
        filename = os.path.basename(image_path).replace("temp_", "result_")
        result_path = os.path.join(self.result_dir, filename)
        cv2.imwrite(result_path, image)
        self.get_logger().info(f"Saved result image: {result_path}")
        
        # 删除临时图片
        os.remove(image_path)

def main(args=None):
    rclpy.init(args=args)
    node = RecognitionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
