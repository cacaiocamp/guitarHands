import cv2
import numpy as np

def update_threshold(val):
    min_val = cv2.getTrackbarPos('Min Threshold', 'Threshold Adjustment')
    max_val = cv2.getTrackbarPos('Max Threshold', 'Threshold Adjustment')
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        return

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Normalize the grayscale frame
    gray_frame_normalized = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX)

    # Apply the thresholding
    _, depth_thresh = cv2.threshold(gray_frame_normalized, min_val, max_val, cv2.THRESH_BINARY)
    
    # Show the thresholded image
    cv2.imshow('Threshold Adjustment', depth_thresh)

# Open the camera
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create a window named 'Threshold Adjustment'
cv2.namedWindow('Threshold Adjustment')

# Create trackbars for adjusting the min and max thresholds
cv2.createTrackbar('Min Threshold', 'Threshold Adjustment', 0, 255, update_threshold)
cv2.createTrackbar('Max Threshold', 'Threshold Adjustment', 255, 255, update_threshold)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Normalize the grayscale frame
    gray_frame_normalized = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX)

    # Get current positions of the trackbars
    min_val = cv2.getTrackbarPos('Min Threshold', 'Threshold Adjustment')
    max_val = cv2.getTrackbarPos('Max Threshold', 'Threshold Adjustment')

    # Apply the thresholding
    _, depth_thresh = cv2.threshold(gray_frame_normalized, min_val, max_val, cv2.THRESH_BINARY)

    # Display the resulting frame
    cv2.imshow('Threshold Adjustment', depth_thresh)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()