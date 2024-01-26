import cv2
import numpy as np

# webcam
cap = cv2.VideoCapture(0) 


params = cv2.SimpleBlobDetector_Params()

params.filterByColor = True
params.blobColor = 0  # 0 for dark blobs, 255 for light blobs

#circularity for rectangles
params.filterByCircularity = True
params.minCircularity = 0.6 

#convexity completely covered
params.filterByConvexity = True
params.minConvexity = 0.9

#inertia ratio for rectangles
params.filterByInertia = True
params.minInertiaRatio = 0.6 

detector = cv2.SimpleBlobDetector_create(params)

while True:
    #webcam
    ret, frame = cap.read()

    #frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #green color range
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    #mask for the green color
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    #AND operation to get the green regions
    green_regions = cv2.bitwise_and(frame, frame, mask=green_mask)

    gray_frame = cv2.cvtColor(green_regions, cv2.COLOR_BGR2GRAY)

    #perform blob detection
    keypoints = detector.detect(gray_frame)

    if keypoints:
        keypoint = keypoints[0]
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        size_percent = (keypoint.size / (frame.shape[0] * frame.shape[1])) * 100

        # the number of white pixels in the blob
        num_white_pixels = np.sum(green_mask[y:y+int(keypoint.size), x:x+int(keypoint.size)] == 255)

        # total number of pixels in the blob region
        total_pixels_in_blob = int(keypoint.size) * int(keypoint.size)

        # ratio of white pixels to total pixels
        ratio_white_to_total = num_white_pixels / total_pixels_in_blob

        print(f"Blob detected at (x={x}, y={y}), "
              f"Ratio of White Pixels to Total Pixels: {ratio_white_to_total}")


    # Show the masked video (black and white)
    cv2.imshow("Masked Video", gray_frame)


    key = cv2.waitKey(1)
    if key == 27:  # 27 corresponds to the 'Esc' key
        break

cap.release()
cv2.destroyAllWindows()
