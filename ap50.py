
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


def calc_iou(pred_box, gt_box):
  #算交集
  x1 = max(pred_box[0], gt_box[0])
  y1 = max(pred_box[1], gt_box[1])
  x2 = min(pred_box[2], gt_box[2])
  y2 = min(pred_box[3], gt_box[3])
  intersection = max(0, x2 - x1) * max(0, y2 - y1)
 
  #算聯集
  pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
  gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
  union = pred_area + gt_area - intersection
 
  #算iou
  iou = intersection / union
  return iou

def calc_ap50(pred_boxes, pred_classes, gt_boxes, gt_classes):
  total_pred = 0
  correct_pred = 0
  for i in range(min(len(pred_boxes), len(gt_boxes))):
    #nms
    nms_pred_boxes = []
    nms_pred_classes = []
    for j in range(len(pred_boxes[i])):
      if pred_boxes[i][j][3] - pred_boxes[i][j][1] > 0 and pred_boxes[i][j][2] - pred_boxes[i][j][0] > 0:
        nms_pred_boxes.append(pred_boxes[i][j])
        nms_pred_classes.append(pred_classes[i][j])
 
    for j in range(len(nms_pred_boxes)):
      max_iou = 0
      best_gt = -1
      for k in range(len(gt_boxes[i])):
        iou = calc_iou(nms_pred_boxes[j], gt_boxes[i][k])
        if iou > max_iou:
          max_iou = iou
          best_gt = k
      total_pred += 1
      if max_iou > 0.5 and nms_pred_classes[j] == gt_classes[i][best_gt]:
        correct_pred += 1
 
  #算ap50
  ap50 = correct_pred / total_pred
  return ap50

pred_boxes = []  # list of lists of predicted bounding boxes for each image
pred_labels = []  # list of lists of predicted labels for each image
gt_boxes = []  # list of lists of ground-truth bounding boxes for each image
gt_labels = [] 
def get_ap50(directory):
        
        cfg = get_cfg()
        microcontroller_metadata = { "name": "category_test", "thing_classes": ["powder_uncover", "powder_uneven", "scratch"]}
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = "/Users/david/Desktop/DIP_final/outputModels1/model_0029999.pth"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = 'cpu'
        cfg.DATASETS.TEST = ("category_test", )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        predictor = DefaultPredictor(cfg) 

        pred_boxes = []  
        pred_labels = []  
        gt_boxes = []  
        gt_labels = [] 
        for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
                json_file = os.path.join(directory, filename)
                img = cv2.imread(json_file.replace('label','image')[:-5] + '.png')
                
                #resize to 1000*1000
                resized_image = cv2.resize(img, (480, 480))  
                #get resize ratio      
                resize_ratio = resized_image.shape[0] / img.shape[0]
                #print(resize_ratio)
                outputs = predictor(img)#predicted outputs
                with open(json_file, "r") as f:#get real json's coordinate
                        data = json.load(f)
                annos = data["shapes"]
                ps = []
                gt_class = []
                for anno in annos:
                        px = [a[0] for a in anno['points']] # x coord
                        py = [a[1] for a in anno['points']] # y-coord
                        poly = [(x, y) for x, y in zip(px, py)] # poly for segmentation
                        poly = [p for x in poly for p in x]
                        ps.append(poly)
                        classes = ['powder_uncover', 'powder_uneven', 'scratch']
                        gt_class.append(classes.index(anno['label']))
        
                #print(gt_class)
                #print(json_file)
                #print(f"\n{ps}\n")
                ps = sorted(ps, key=lambda x: x[0])#get real json's coordinate
                for list in ps:
                        if list[1] > list[3]:
                                list[1], list[3] = list[3], list[1]
                out_iou = outputs["instances"].pred_classes#sort cuz it's in wrong order
                pred_box = outputs["instances"].pred_boxes.tensor.cpu()
                pred_box = [t.tolist() for t in pred_box]
                pred_class = [t.tolist() for t in out_iou]
                
                pred_boxes.append(pred_box)
                pred_labels.append(pred_class)
                gt_boxes.append(ps)
                gt_labels.append(gt_class)
        return(calc_ap50(pred_boxes, pred_labels, gt_boxes, gt_labels))