yolo_path = '/project/train/src_repo/v5/yolov5'
model_path = '/project/train/models/*/weights/best.pt'
names = ['fire','big_fire','smoke']
conf_thres = 0.3
iou_thres = 0.05
prob_thres = 0.3
imgsz = 480

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import sys
import json
sys.path.insert(1, yolo_path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import argparse
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from utils.augmentations import letterbox
from models.experimental import attempt_load
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync
from glob import glob


model_path=glob(model_path)[0]
device = '0'
stride = 32


def init():
    # Initialize
    global imgsz, device, stride,model_path
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.jit.load(model_path) if 'torchscript' in model_path else attempt_load(model_path, device=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    model.eval()
    model.half()  # to FP16
    return model

@torch.no_grad()
def process_image(model, input_image=None, args=None, **kwargs):
    # Padded resize
    img0 = input_image
    img = letterbox(img0, new_shape=imgsz, stride=stride, auto=True)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half()
#     img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None] 
    pred = model(img, augment=False)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)

    fake_result = {}

    fake_result["algorithm_data"] = {
       "is_alert": False,
       "target_count": 0,
       "target_info": []
   }
    fake_result["model_data"] = {"objects": []}
    # Process detections
    cnt = 0
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                if conf < prob_thres:
                    continue
                cnt += 1
                fake_result["model_data"]['objects'].append({
                    "x": int(xyxy[0]),
                    "y": int(xyxy[1]),
                    "width": int(xyxy[2])-int(xyxy[0]),
                    "height": int(xyxy[3])-int(xyxy[1]),
                    "confidence":float(conf),
                    "name":names[int(cls)]
                })
                fake_result["algorithm_data"]["target_info"].append({
                    "x": int(xyxy[0]),
                    "y": int(xyxy[1]),
                    "width": int(xyxy[2])-int(xyxy[0]),
                    "height": int(xyxy[3])-int(xyxy[1]),
                    "confidence":float(conf),
                    "name":names[int(cls)]
                }
                )
    if cnt:
        fake_result ["algorithm_data"]["is_alert"] = True
        fake_result ["algorithm_data"]["target_count"] = cnt
    return json.dumps(fake_result, indent = 4)


if __name__ == '__main__':
    from glob import glob
    # Test API
    
    image_names = glob('/home/data/*/*.jpg')[:10]
    predictor = init()
    s = 0
    for image_name in image_names:
        img = cv2.imread(image_name)
        t1 = time.time()
        res = process_image(predictor, img)
        print(res)
        t2 = time.time()
        s += t2 - t1
    print(1/(s/100))
