import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--videoPath', nargs='?', default="E:\\output\\Data\\Group_09-master.mkv")
parser.add_argument('--outputFolder', nargs='?', default="E:\\output\\Data\\Group9Calibration")
parser.add_argument('--videoName', nargs='?', default="Group9-master")
parser.add_argument('--initialFrame', nargs='?', default=130)

args = parser.parse_args()
calibrated = False

if not os.path.isdir(args.outputFolder):
    os.mkdir(args.outputFolder)

# file
cap = cv2.VideoCapture(args.videoPath)
cap.set(cv2.CAP_PROP_POS_FRAMES, args.initialFrame)

_, frame = cap.read()
frame = cv2.resize(frame, (960, 540)) 
h, w, c = frame.shape
frameCount = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.resize(frame, (960, 540)) 
    frameCount+=1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.putText(frame, "Frame: " + str(frameCount + args.initialFrame), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey() == ord('q'):
        break

    if cv2.waitKey() == ord('s'):
         cv2.imwrite(f"{args.outputFolder}\\{str(args.videoName)}_{str(frameCount)}.png", frame)
       
# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()