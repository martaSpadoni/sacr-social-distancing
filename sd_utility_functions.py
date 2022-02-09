import numpy as np

def resize_box(box, h, w):
  return tuple(np.array([box[0]*h, box[1]*w, box[2]*h, box[3]*w]).astype(int))

def area(rect):
  return (rect[0] - rect[2]) * (rect[1] - rect[3])

def iou(label_box, pred_box):
  pred_area = area(pred_box)
  label_area = area(label_box)

  ix = min(label_box[3], pred_box[3]) - max(label_box[1], pred_box[1])
  iy = min(label_box[2], pred_box[2]) - max(label_box[0], pred_box[0])

  ix = max(ix, 0)
  iy = max(iy, 0)

  intersection = ix * iy

  union_area = pred_area + label_area - intersection
  union_area = max(union_area, np.finfo(float).eps)

  return intersection / union_area

def find_corresponding_bbox(pred_box, label_boxes):
  ious = np.array([iou(label_box, pred_box) for label_box in label_boxes])
  return np.argmax(ious), np.amax(ious)

def compare_boxes(pred_boxes, label_boxes):
  result = [find_corresponding_bbox(pred_box, label_boxes) for pred_box in pred_boxes]
  return result