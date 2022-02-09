import numpy as np
import math
import cv2

TARGET_W = 800
TARGET_H = 450

def compute_perspective_matrix(upper_left, upper_right, bottom_left, bottom_right, image):
  corners = np.float32([upper_left, upper_right, bottom_left, bottom_right])
  h, w = TARGET_H, TARGET_W
  params = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

  matrix = cv2.getPerspectiveTransform(corners, params)
  transformation = cv2.warpPerspective(image, matrix, (w,h))
  return matrix, transformation

def perspective_transform(matrix, image):
  h, w = TARGET_H, TARGET_W
  return cv2.warpPerspective(image, matrix, (w,h))

def get_bbox_centroid(box):
  y1, x1, y2, x2 = box[0], box[1], box[2], box[3]
  #return [(x1+x2)/2, (y1+y2)/2]
  return ((x1+x2)/2, y2)
  
def map_centroid(centroid, matrix):
  return cv2.perspectiveTransform(centroid, matrix)

def compute_bbox_centroids(detection_boxes):
  return [get_bbox_centroid(box) for box in detection_boxes]
  

def compute_bird_eye_centroids(detection_boxes, matrix):
  detection_centroids = compute_bbox_centroids(detection_boxes)
  mapped_centroids = [map_centroid(np.array([np.array([centroid])]), matrix) for centroid in detection_centroids]
  mapped_centroids = [(int(centroid[0][0][0]), int(centroid[0][0][1])) for centroid in mapped_centroids]
  return mapped_centroids
