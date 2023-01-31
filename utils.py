
import PySimpleGUI as sg
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
import json
import cv2
from PIL import Image, ImageTk
import io
import os
import os.path
import time
import numpy as np
from keras import backend as K

def euclid(v1, v2):
    distance = 0
    for i in range(len(v1)):
        distance += (v1[i] - v2[i])**2
    return distance / len(v1)

def calculate_similarity_gt(box1, box2):#ps比out_iou長 存回ps 若相反則import時倒轉
    new_box = []
    for index2, box2_item in enumerate(box2):
        similarity = []
        for index1, box1_item in enumerate(box1):
            similarity.append(euclid(box2_item, box1_item))
        new_box.append(box1[similarity.index(min(similarity))])
    new_box = [tuple(b) for b in new_box]
    return list(set(new_box))

def calculate_iou(box1, box2):
    # Calculate area of each bounding box
    x1, y1, x2, y2 = box1
    box1_area = abs((x2 - x1) * (y2 - y1))
    x1, y1, x2, y2 = box2
    box2_area = abs((x2 - x1) * (y2 - y1))
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    #if x1 < x2 and y1 < y2:
    intersection_area = abs((x2 - x1) * (y2 - y1))

    
    union_area = box1_area + box2_area - intersection_area

    if(union_area < 0):return 0

    # Return IOU
    return intersection_area / union_area

def IOU_file(filename): 
        cfg = get_cfg()
        microcontroller_metadata = { "name": "category_test", "thing_classes": ["powder_uncover", "powder_uneven", "scratch"]}
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = "/Users/david/Desktop/DIP_final/outputModels1/model_final.pth"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = 'cpu'
        cfg.DATASETS.TEST = ("category_test", )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        predictor = DefaultPredictor(cfg) 
        json_file = filename
        json_file = json_file.replace("label", "image")
        img = cv2.imread(json_file[:-5] + '.png')
        outputs = predictor(img)#predicted outputs
        json_file = json_file.replace("image", "label")
        with open(json_file, "r") as f:#get real json's coordinate
            data = json.load(f)
            annos = data["shapes"]
            ps = []
            for anno in annos:
                    px = [a[0] for a in anno['points']] # x coord
                    py = [a[1] for a in anno['points']] # y-coord
                    poly = [(x, y) for x, y in zip(px, py)] # poly for segmentation
                    poly = [p for x in poly for p in x]
                    ps.append(poly)
        #print(ps)
        ps = sorted(ps, key=lambda x: x[0])#get real json's coordinate
        out_iou = outputs["instances"].pred_boxes.tensor.cpu()#sort cuz it's in wrong order
        #print(out_iou)
        list_of_lists = [t.tolist() for t in out_iou]
        list_of_lists = sorted(list_of_lists, key=lambda x:x[0])
        
        
        
        iou = 0
        if(len(ps) < len(list_of_lists)):
            list_of_lists = calculate_similarity_gt(list_of_lists, ps)
            list_of_lists = sorted(list_of_lists, key=lambda x:x[0])
            ps = sorted(ps, key=lambda x: x[0])
            for i, b1 in enumerate(list_of_lists):
                b2 = ps[i]
                iou += calculate_iou(b1, b2)
                #print(f'IOU for boxes {i}: {iou}')
            iou = iou / len(list_of_lists)
        elif(len(ps) > len(list_of_lists)):
            ps = calculate_similarity_gt(ps, list_of_lists)
            list_of_lists = sorted(list_of_lists, key=lambda x:x[0])
            ps = sorted(ps, key=lambda x: x[0])
            for i, b1 in enumerate(ps):
                b2 = list_of_lists[i]
                iou += calculate_iou(b1, b2)
                #print(f'IOU for boxes {i}: {iou}')
            iou = iou / len(ps)
        else:
            list_of_lists = sorted(list_of_lists, key=lambda x:x[0])
            ps = sorted(ps, key=lambda x: x[0])
            for i, b1 in enumerate(ps):
                b2 = list_of_lists[i]
                iou += calculate_iou(b2, b1)
                #print(f'IOU for boxes {i}: {iou}')
            iou = iou / len(ps)

        if(iou > 1): iou = 1#有些bug>1我抓不到
            
        return iou
        #print(list_of_lists)
        #print(ps)
def dice_coef(y_true, y_pred):
    y_true[y_true > 0] = 1
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    dice_score = 2 * intersection / union
    return dice_score.numpy()



if __name__ == "main":

    print("fuck")