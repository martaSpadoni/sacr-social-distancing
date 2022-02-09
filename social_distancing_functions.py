import math
import cv2
import itertools as iter
import numpy as np

from perspective_functions import compute_bbox_centroids


def compute_safe_distance_in_px(real_width, target_width = 800, SAFE_DISTANCE_THR_METER = 1.5):
  return (target_width * SAFE_DISTANCE_THR_METER)/real_width
  #return 110

def distance_between(p1, p2):
  return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def check_centroids_distancing(frame_centroids, real_width):
  checked_centroids = []
  mask = []
  couples = [couple for couple in iter.combinations(frame_centroids, 2)]
  for couple in couples:
      distance = distance_between(couple[0], couple[1])
      checked_centroids.append((couple, distance <= compute_safe_distance_in_px(real_width)))
      mask.append(distance <= compute_safe_distance_in_px(real_width))
  
  return checked_centroids, mask

def check_boxes_distancing(frame_boxes, mask):
    checked_boxes = []
    red = []
    couples = [couple for couple in iter.combinations(frame_boxes, 2)]
    for i, couple in enumerate(couples):
        if mask[i]:
            red.append((couple[0][0], couple[0][1], couple[0][2], couple[0][3]))
            red.append((couple[1][0], couple[1][1], couple[1][2], couple[1][3]))
    red_boxes_in_frame = list(set(red))
    for b in frame_boxes:
        checked_boxes.append((b, (0, 0, 255) if b in red_boxes_in_frame else (0, 255, 0)))
    return checked_boxes

def draw_centroids_on_frame(frame, frame_centroids, color = (0,0,255), with_social_distancing = False):
  for centroid in frame_centroids:
    centroid = (int(centroid[0]), int(centroid[1]))
    frame = cv2.circle(frame, centroid, 4, color, -1)
    if with_social_distancing:
        frame = cv2.circle(frame, centroid, 50, color, 2)
  return frame

def draw_centroids_on_bird_eye_frame(bird_eye_centroids):
  #frame = np.zeros((450,800,3),dtype=np.uint8)
  frame = cv2.imread("bird_eye_frame.png")
  if frame is None:
    print("ERROR")
  red = []
  green = []
  for c in bird_eye_centroids:
    if c[1]:
      red.append(c[0][0])
      red.append(c[0][1])
    else :
        if c[0][0] not in red:
            green.append(c[0][0])
        if c[0][1] not in red:
            green.append(c[0][1])
  frame = draw_centroids_on_frame(frame, list(set(red)), with_social_distancing=True)
  frame = draw_centroids_on_frame(frame, list(set(green).difference(red)), (0, 255, 0), True)
  return frame

def draw_checked_boxes_on_frame(frame, checked_boxes):
  _img = frame.copy()
  for b in checked_boxes:
    start_point = (b[0][1], b[0][0])
    end_point = (b[0][3], b[0][2])
    cv2.rectangle(_img, start_point, end_point, b[1], 1)
  return _img


def draw_centroids_and_line_on_frame(frame_boxes, mask):
  frame = np.zeros((450,800,3),dtype=np.uint8) 
  centroids = []
  for c in compute_bbox_centroids(frame_boxes):
    centroids.append((int(c[0]), int(c[1])))
  red = []
  couples = [couple for couple in iter.combinations(centroids, 2)]
  for i, couple in enumerate(couples):
    if mask[i]:
      red.append(couple[0])
      red.append(couple[1])
      frame = cv2.circle(frame, couple[0], 6, (0,0,255), -1)
      frame = cv2.circle(frame, couple[1], 6, (0,0,255), -1)
      frame = cv2.line(frame, couple[0], couple[1], (0,0,255), 4)

  green = set(centroids).difference(set(red))
  for g in green:
    frame = cv2.circle(frame, g, 4, (0,255, 0), -1)
  
  return frame