# based off example on this blog: https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()
parser.add_argument('--videoPath', nargs='?', default="E:\\output\\Data\\Group_09-sub2.mkv")
parser.add_argument('--jsonPath', nargs='?', default="E:\\output\\Data\\Group_09-sub2.json")
parser.add_argument('--initialFrame', nargs='?', default=130)

args = parser.parse_args()
calibrated = False

# file
cap = cv2.VideoCapture(args.videoPath)
cap.set(cv2.CAP_PROP_POS_FRAMES, args.initialFrame)
jsonFile = open(args.jsonPath)

skeletonData = json.load(jsonFile)
frameData = skeletonData["frames"]

_, frame = cap.read()
frame = cv2.resize(frame, (960, 540)) 
h, w, c = frame.shape
frameCount = 0

#criteria used by checkerboard pattern detector.
#Change this if the code can't find the checkerboard
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
rows = 6 #number of checkerboard rows.
columns = 9 #number of checkerboard columns.
world_scaling = 0.015 #change this to the real world square size. Or not.
 
#coordinates of squares in the checkerboard world space
objp = np.zeros((rows*columns,3), np.float32)
objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
objp = world_scaling* objp
 
#Pixel coordinates of checkerboards
imgpoints = [] # 2d points in image plane.
 
#coordinates of the checkerboard in checkerboard world space.
objpoints = [] # 3d point in real world space

while cap.isOpened():
    success, frame = cap.read()
    frame = cv2.resize(frame, (960, 540)) 
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frameCount+=1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if calibrated:
        #frame = cv2.undistort(frame, cameraMatrix, dist, None, None)
        bodies = frameData[args.initialFrame + frameCount]["bodies"]
        for body in bodies:
            for joint in body["joint_positions"]:
                points_2d, _ = cv2.projectPoints(np.array(joint), np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), cameraMatrix, dist)
                print(points_2d)
                print("width " + str(w) + " height " + str(h))
                shift = 7
                cv2.circle(frame, (int(points_2d[0][0][0] * 2**shift),int(points_2d[0][0][1] * 2**shift)), radius=5, color=(0, 0, 255), thickness=5, shift=shift)

    if not calibrated:
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        GRID = (rows, columns)
        ret, corners = cv2.findChessboardCorners(gray, GRID, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            # refining pixel coordinates for given 2d points.
            # https://theailearner.com/tag/cv2-cornersubpix/ - window size
            corners2 = cv2.cornerSubPix(gray, corners, (1,1), (-1,-1), criteria)
            
            imgpoints.append(corners2)
            objpoints.append(objp)
    
            # Draw and display the corners
            img = cv2.drawChessboardCorners(frame, GRID, corners2, ret)


    cv2.putText(frame, "Frame: " + str(frameCount + args.initialFrame), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(10) == ord('q'):
        break

    if frameCount % 20 == 0 and not calibrated:
        ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
        print(ret)
        print(cameraMatrix)
        print(dist)
        print(rvecs)
        print(tvecs)
        calibrated = True
       
# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()