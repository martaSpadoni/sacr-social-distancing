import numpy as np
import cv2
from sd_utility_functions import iou, resize_box

def non_max_suppression(boxes, scores, classes, IOU_THR = 0.35):
  non_max = []
  for i in range(len(boxes)):
    for j in range(i+1, len(boxes)):
      if iou(boxes[i], boxes[j]) > IOU_THR:
        non_max.append(i if scores[i] < scores[j] else j)
  
  non_max = list(set(non_max))
  non_max.sort(reverse=True)

  for index in non_max:
    classes = np.delete(classes, index, 0)
    scores = np.delete(scores, index, 0)
    boxes = [tuple(b) for b in np.delete(boxes, index, 0)]
  return classes, scores, boxes

def detect(model, _image, CONF_THR = 0.37):
  image = _image[np.newaxis, ...].copy()
  h,w,_ = _image.shape
  results = model(image)
  result = {key:value.numpy() for key,value in results.items()}
  classes = result["detection_classes"].astype(int)
  scores = result["detection_scores"]
  mask = np.logical_and(scores > CONF_THR, classes == 1)
  scores = scores[mask]
  boxes = result["detection_boxes"][mask]
  boxes = [resize_box(box, h, w) for box in boxes]
  classes = classes[mask]

  return non_max_suppression(boxes, scores, classes)

def draw_boxes_on_frame(frame, boxes):
  _img = np.array(frame)
  for box in boxes:
    start_point = (box[1], box[0])
    end_point = (box[3], box[2])
    cv2.rectangle(_img, start_point, end_point, (255, 0, 0), 1)
  return _img

def box_frames(frames, boxes):
  return [draw_boxes_on_frame(f, b) for f,b in zip(frames, boxes)]
