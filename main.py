import cv2
import numpy as np

# Function to capture user movement
def capture_movement():
    # Open the default camera (0) or you can specify another camera if you have multiple
    cap = cv2.VideoCapture(0)
    
    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=200,  # Increase the number of corners
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(21, 21),  # Increase window size for better accuracy
                     maxLevel=3,  # Increase the number of pyramid layers
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Read the first frame
    ret, frame1 = cap.read()
    
    if not ret:
        print("Error: Unable to read the first frame.")
        cap.release()
        return
    
    # Convert the first frame to grayscale
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    # Detect initial points to track
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    
    if prev_pts is None:
        print("Error: No features found to track.")
        cap.release()
        return
    
    # Create a mask image for drawing (optional, for visualization purposes)
    mask = np.zeros_like(frame1)
    
    while True:
        # Read the next frame
        ret, frame2 = cap.read()
        
        if not ret:
            break
        
        # Convert the current frame to grayscale
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow using Lucas-Kanade method
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
        
        if next_pts is None:
            print("Warning: Lost track of features, re-detecting.")
            prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            if prev_pts is None:
                print("Error: No features found to track.")
                break
            prev_gray = gray.copy()
            continue
        
        # Select good points
        good_new = next_pts[status == 1]
        good_old = prev_pts[status == 1]
        
        # Draw the movement vectors
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a, b, c, d = int(a), int(b), int(c), int(d)
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            frame2 = cv2.circle(frame2, (a, b), 5, (0, 0, 255), -1)
        
        # Combine the original frame with the mask
        output = cv2.add(frame2, mask)
        
        # Display the frame with movement vectors
        cv2.imshow("User Movement", output)
        
        # Update the previous frame and previous points
        prev_gray = gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to capture movement
capture_movement()
