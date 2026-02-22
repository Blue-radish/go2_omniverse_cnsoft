# -*- coding: utf-8 -*-
import codecs
import os
import time
import sys
sys.path.append('PaddleDetection')
import json
import yaml
from functools import reduce
import multiprocessing

from PIL import Image
import cv2
import numpy as np
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

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

def get_test_images(infer_file):
    infer_dir = os.path.dirname(os.path.abspath(infer_file))
    with open(infer_file, 'r') as f:
        dirs = f.readlines()
    images = []
    for line in dirs:
        line = line.strip()
        if line:
            line = line.replace('\\', '/')
            if not os.path.isabs(line):
                line = os.path.join(infer_dir, line)
            images.append(line)
    assert len(images) > 0, "no image found in {}".format(infer_file)
    return images

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

def predict_image(detector, image_list, result_path, threshold):
    c_results = {"result": []}
    for im_path in image_list:
        input_im_lst = []
        input_im_info_lst = []
        im, im_info = preprocess(im_path, detector.preprocess_ops)
        input_im_lst.append(im)
        input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        image_id = os.path.basename(im_path).split('.')[0]
        det_results = detector.predict(inputs)
        im_bboxes_num = det_results['boxes_num'][0]
        if im_bboxes_num > 0:
            bbox_results = det_results['boxes'][0:im_bboxes_num, 2:]
            id_results = det_results['boxes'][0:im_bboxes_num, 0]
            score_results = det_results['boxes'][0:im_bboxes_num, 1]
            for idx in range(im_bboxes_num):
                if float(score_results[idx]) >= threshold:
                    c_results["result"].append({
                        "image_id": image_id,
                        "type": int(id_results[idx]) + 1,
                        "x": float(bbox_results[idx][0]),
                        "y": float(bbox_results[idx][1]),
                        "width": float(bbox_results[idx][2]) - float(bbox_results[idx][0]),
                        "height": float(bbox_results[idx][3]) - float(bbox_results[idx][1]),
                        "segmentation": []
                    })
    try:
        print(f"Writing results to {result_path}")
        with open(result_path, 'w') as ft:
            json.dump(c_results, ft)
        print(f"Results written successfully to {result_path}")
    except Exception as e:
        print(f"Error writing results to {result_path}: {e}")

def main(infer_txt, result_path, det_model_path, threshold):
    pred_config = PredictConfig(det_model_path)
    detector = Detector(pred_config, det_model_path)
    img_list = get_test_images(infer_txt)
    predict_image(detector, img_list, result_path, threshold)

if __name__ == '__main__':
    start_time = time.time()
    det_model_path = "model/"
    threshold = 0.3
    infer_txt = sys.argv[1]
    result_path = sys.argv[2]
    print(f"Using result path: {result_path}")
    main(infer_txt, result_path, det_model_path, threshold)
    print('total time:', time.time() - start_time)