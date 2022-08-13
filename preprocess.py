import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2
import logging
from glob import glob
import time
import random

'''
YOLO v5 
xml -> txt
'''

classes = ['fixed_stall','sunshade','drying_object']
class2id = {name:i for i, name in enumerate(classes)}

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])

    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    x = min(x, 1.0)
    y = min(y, 1.0)
    w = min(w, 1.0)
    h = min(h, 1.0)
    
    return (x,y,w,h)
 
def convert_annotation(image_path):
    in_file = open(image_path.replace('.xml', '.xml'),encoding="utf-8")
    out_file = open(image_path.replace('.xml', '.txt'), 'w')
    # print(in_file)
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    if w == 0 or h == 0: 
        print(1)
        return

    for obj in root.iter('object'):
        name = obj.find('name').text
        cls_id = class2id[name]
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b,)

# 1. 转换数据label
files = glob('/home/data/*/*.xml')
for file in files:
    convert_annotation(file)

# 2. 划分train与valid ~ K折
K = 5
files = glob('/home/data/*/*.txt')
random.shuffle(files)
ind = len(files) // 5
train = [x.replace('.txt', '.jpg')+'\n' for x in files[ind:]]
valid = [x.replace('.txt', '.jpg')+'\n' for x in files[:ind]]
print(f"train {len(train)}, valid {len(valid)}")

# 3. 写入文件
with open('train.txt', 'w') as f:
    f.writelines(train)
with open('valid.txt', 'w') as f:
    f.writelines(valid)
