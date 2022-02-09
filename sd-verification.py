import numpy as np
import tensorflow as tf
import cv2
from detection import detect
from perspective_functions import compute_bbox_centroids, compute_bird_eye_centroids, perspective_transform
from social_distancing_functions import *
from video_utilities import *
import time

#Requirements
# - SSD-MobileNet v2: run the following commmands to download the model (it must be saved in the same directory of this script)
# wget "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
# tar -xvzf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
# rm ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
# - Perspective Matrix to compute the bird-eye model of the scene: run 'python sd-matrix-py' to compute and save it as file
video_name = "room5-standing-mini"
REAL_ROOM_WIDTH = 9
#Loading the model
model = tf.saved_model.load("ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model")
#Loading the perspective matrix 
matrix = np.load("perspective_matrix.npy")
#Loading the frame on which the social distancing must be checked 
#frame = cv2.imread("room4-standing-moment.jpg")
#frame = cv2.resize(frame, (800, 450))
frames = get_video_frames(video_name+".mp4")

simulation_frames = []
boxed_frames = []
lines_frames = []
print("Inference started...")
for frame in frames:
    start_time = time.time()
    _, _, detection_boxes= detect(model, frame)
    end_time = time.time()
    #print("Inference time: ", end_time - start_time)
    bird_eye_centroids = compute_bird_eye_centroids(detection_boxes, matrix)
    checked_be_centroids, mask = check_centroids_distancing(bird_eye_centroids, REAL_ROOM_WIDTH)
    frame_centroids = compute_bbox_centroids(detection_boxes)
    checked_frame = draw_checked_boxes_on_frame(frame, check_boxes_distancing(detection_boxes, mask))
    boxed_frames.append(draw_centroids_on_frame(checked_frame, frame_centroids))
    simulation_frames.append(draw_centroids_on_bird_eye_frame(checked_be_centroids))
    lines_frames.append(draw_centroids_and_line_on_frame(detection_boxes, mask))
print("Inference ended, saving videos...")

if len(boxed_frames) > 0:
   save_video(boxed_frames, video_name+"-boxed.avi") 
if len(simulation_frames) > 0:
   save_video(simulation_frames, video_name+"-simulation.avi")
if len(lines_frames) > 0:
   save_video(lines_frames, video_name+"-lines.avi")
    
