import cv2
import numpy as np
from perspective_functions import compute_perspective_matrix

# Constants
WIDTH = 800
HEIGHT = 450
TARGET_SIZE = (WIDTH, HEIGHT)

# Create an empty list of points for the coordinates
list_points = list()
 
# Define the callback function that we are going to use to get our coordinates
def CallBackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button of the mouse is clicked - position (", x, ", ",y, ")")
        list_points.append([x,y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("Right button of the mouse is clicked - position (", x, ", ", y, ")")
        list_points.append([x,y])




# Create a black image and a window
windowName = 'Scene Settings'
cv2.namedWindow(windowName)

#TODO: use the picamera for take automatically the initial frame on which will be detect the corners

# Load the image 
img_path = "room5-init-frame.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, TARGET_SIZE)

# Get the size of the image for the calibration
width,height,_ = img.shape


# bind the callback function to window
cv2.setMouseCallback(windowName, CallBackFunc)


if __name__ == "__main__":
    # Check if the 4 points have been saved
    while (True):
        cv2.imshow(windowName, img)
        if len(list_points) == 4:
            pts = np.array(list_points)
            print(pts)
            #_img = cv2.drawContours(img, [pts], 0, (255, 0, 0), 3)
            #cv2.imwrite("image_rect.png", _img)
            matrix, bird_eye_frame = compute_perspective_matrix(pts[0], pts[1], pts[2], pts[3], img)
            cv2.imwrite("bird_eye_frame.png", bird_eye_frame)
            print(matrix)
            np.save("perspective_matrix.npy", matrix)
            break
        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()