#https://aliyasineser.medium.com/opencv-camera-calibration-e9a48bdd1844

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import glob

def calibrate(dirpath, image_format, square_size, width=9, height=6):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size # if square_size is 1.5 centimeters, it would be better to write it as 0.015 meters. Meter is a better metric because most of the time we are working on meter level projects.
    
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Some people will add "/" character to the end. It may brake the code so I wrote a check.
    if dirpath[-1:] == '/':
        dirpath = dirpath[:-1]
    images = glob.glob(dirpath+'/' + '*.' + image_format) #

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return [ret, mtx, dist, rvecs, tvecs]

parser = argparse.ArgumentParser(description='Camera calibration')
parser.add_argument('--image_dir', nargs='?', help='image directory path', default="E:\\output\\Data\\Group9Calibration\\")
parser.add_argument('--image_format', nargs='?',  help='image format, png/jpg', default='png')
parser.add_argument('--square_size', nargs='?', help='chessboard square size', default=1)
parser.add_argument('--width', nargs='?', help='chessboard width size, default is 9', default=9)
parser.add_argument('--height', nargs='?', help='chessboard height size, default is 6', default=6)
parser.add_argument('--videoPath', nargs='?', default="E:\\output\\Data\\Group_09-sub2.mkv")
parser.add_argument('--jsonPath', nargs='?', default="E:\\output\\Data\\Group_09-sub2.json")
parser.add_argument('--initialFrame', nargs='?', default=130)

args = parser.parse_args()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ret, mtx, dist, rvecs, tvecs = calibrate(args.image_dir, args.image_format, args.square_size, args.width, args.height)

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

while cap.isOpened():
    success, frame = cap.read()
    frame = cv2.resize(frame, (960, 540)) 
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frameCount+=1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bodies = frameData[args.initialFrame + frameCount]["bodies"]
    for body in bodies:
        for joint in body["joint_positions"]:
            points_2d, _ = cv2.projectPoints(np.array(joint), rvecs[0], tvecs[0], mtx, dist)
            print(points_2d)
            print("width " + str(w) + " height " + str(h))
            shift = 7

            try:
                cv2.circle(frame, (int(points_2d[0][0][0]), int(points_2d[0][0][1])), radius=5, color=(0, 0, 255), thickness=5)
            except Exception:
               print("error rendering point")


    cv2.putText(frame, "Frame: " + str(frameCount + args.initialFrame), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(10) == ord('q'):
        break
       
# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()