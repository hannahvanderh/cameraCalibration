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

# file
cap = cv2.VideoCapture(args.videoPath)
cap.set(cv2.CAP_PROP_POS_FRAMES, args.initialFrame)
jsonFile = open(args.jsonPath)

skeletonData = json.load(jsonFile)
frameData = skeletonData["frames"]
cameraCalibration = skeletonData["camera_calibration"]

if(cameraCalibration != None):
    cameraMatrix = np.array([np.array([float(cameraCalibration["fx"]),0,float(cameraCalibration["cx"])]), 
                             np.array([0,float(cameraCalibration["fy"]),float(cameraCalibration["cy"])]), 
                             np.array([0,0,1])])
    rotation = np.array([
        np.array([float(cameraCalibration["rotation"][0]),float(cameraCalibration["rotation"][1]),float(cameraCalibration["rotation"][2])]), 
        np.array([float(cameraCalibration["rotation"][3]),float(cameraCalibration["rotation"][4]),float(cameraCalibration["rotation"][5])]), 
        np.array([float(cameraCalibration["rotation"][6]),float(cameraCalibration["rotation"][7]),float(cameraCalibration["rotation"][8])])])
    
    translation = np.array([float(cameraCalibration["translation"][0]), float(cameraCalibration["translation"][1]), float(cameraCalibration["translation"][2])])

    dist = np.array([
        float(cameraCalibration["k1"]), 
        float(cameraCalibration["k2"]),
        float(cameraCalibration["p1"]),
        float(cameraCalibration["p2"]),
        float(cameraCalibration["k3"]),
        float(cameraCalibration["k4"]),
        float(cameraCalibration["k5"]),
        float(cameraCalibration["k6"])])

_, frame = cap.read()
h, w, c = frame.shape
frameCount = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frameCount+=1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if cameraCalibration != None:
        bodies = frameData[args.initialFrame + frameCount]["bodies"]
        for body in bodies:
            for joint in body["joint_positions"]:
                points_2d, _ = cv2.projectPoints(
                    np.array(joint), 
                    rotation,
                    translation,
                    cameraMatrix,
                    dist)
                
                print(points_2d)
                shift = 7
                cv2.circle(frame, (int(points_2d[0][0][0] * 2**shift),int(points_2d[0][0][1] * 2**shift)), radius=5, color=(0, 0, 255), thickness=10, shift=shift)

    cv2.putText(frame, "Frame: " + str(frameCount + args.initialFrame), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    frame = cv2.resize(frame, (960, 540))
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(10) == ord('q'):
        break
       
# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()