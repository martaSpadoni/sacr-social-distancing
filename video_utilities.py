import cv2

def get_video_frames(video_path, rotate_right=False, TARGET_W = 800, TARGET_H = 450):
  vid = cv2.VideoCapture(video_path)
  frames = []

  while(vid.isOpened()):
    ret, fr = vid.read()
    if ret == True:
      fr = cv2.resize(fr, (TARGET_W, TARGET_H))
      if rotate_right:
        fr = cv2.rotate(fr, cv2.cv2.ROTATE_90_CLOCKWISE)
      frames.append(fr)
    else:
      break

  vid.release()
  return frames

def save_video(frames, video_path="video.avi", TARGET_W = 800, TARGET_H = 450):
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (TARGET_W, TARGET_H))
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()
