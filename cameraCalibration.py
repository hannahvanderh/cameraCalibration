import argparse
import cv2
import numpy as np
import json
from enum import Enum
import mediapipe as mp

class Joint(Enum):
        PELVIS = 0
        SPINE_NAVEL = 1
        SPINE_CHEST = 2
        NECK = 3
        CLAVICLE_LEFT = 4
        SHOULDER_LEFT = 5
        ELBOW_LEFT = 6
        WRIST_LEFT = 7
        HAND_LEFT = 8
        HANDTIP_LEFT = 9
        THUMB_LEFT = 10
        CLAVICLE_RIGHT = 11
        SHOULDER_RIGHT = 12
        ELBOW_RIGHT = 13
        WRIST_RIGHT = 14
        HAND_RIGHT = 15
        HANDTIP_RIGHT = 16
        THUMB_RIGHT = 17
        HIP_LEFT = 18
        KNEE_LEFT = 19
        ANKLE_LEFT = 20
        FOOT_LEFT = 21
        HIP_RIGHT = 22
        KNEE_RIGHT = 23
        ANKLE_RIGHT = 24
        FOOT_RIGHT = 25
        HEAD = 26
        NOSE = 27
        EYE_LEFT = 28
        EAR_LEFT = 29
        EYE_RIGHT = 30
        EAR_RIGHT = 31

class BodyCategory(Enum):
        HEAD = 0
        RIGHT_ARM = 1
        LEFT_ARM = 2
        TORSO = 3
        RIGHT_LEG = 4
        LEFT_LEG = 4

def getPointSubcategory(joint):
     if(joint == Joint.PELVIS or joint == Joint.NECK or joint == Joint.SPINE_NAVEL or joint == Joint.SPINE_CHEST):
          return BodyCategory.TORSO
     if(joint == Joint.CLAVICLE_LEFT or joint == Joint.SHOULDER_LEFT or joint == Joint.ELBOW_LEFT 
        or joint == Joint.WRIST_LEFT or joint == Joint.HAND_LEFT or joint == Joint.HANDTIP_LEFT or joint == Joint.THUMB_LEFT):
          return BodyCategory.LEFT_ARM
     if(joint == Joint.CLAVICLE_RIGHT or joint == Joint.SHOULDER_RIGHT or joint == Joint.ELBOW_RIGHT 
        or joint == Joint.WRIST_RIGHT or joint == Joint.HAND_RIGHT or joint == Joint.HANDTIP_RIGHT or joint == Joint.THUMB_RIGHT):
          return BodyCategory.RIGHT_ARM
     if(joint == Joint.HIP_LEFT or joint == Joint.KNEE_LEFT or joint == Joint.ANKLE_LEFT or joint == Joint.FOOT_LEFT):
          return BodyCategory.LEFT_LEG
     if(joint == Joint.HIP_RIGHT or joint == Joint.KNEE_RIGHT or joint == Joint.ANKLE_RIGHT or joint == Joint.FOOT_RIGHT):
          return BodyCategory.RIGHT_LEG
     if(joint == Joint.HEAD or joint == Joint.NOSE or joint == Joint.EYE_LEFT 
        or joint == Joint.EAR_LEFT or joint == Joint.EYE_RIGHT or joint == Joint.EAR_RIGHT):
          return BodyCategory.HEAD

bone_list = [
        [
            Joint.SPINE_CHEST, 
            Joint.SPINE_NAVEL
        ],
        [
            Joint.SPINE_NAVEL,
            Joint.PELVIS
        ],
        [
            Joint.SPINE_CHEST,
            Joint.NECK
        ],
        [
            Joint.NECK,
            Joint.HEAD
        ],
        [
            Joint.HEAD,
            Joint.NOSE
        ],
        [
            Joint.SPINE_CHEST,
            Joint.CLAVICLE_LEFT
        ],
        [
            Joint.CLAVICLE_LEFT,
            Joint.SHOULDER_LEFT
        ],
        [
            Joint.SHOULDER_LEFT,
            Joint.ELBOW_LEFT
        ],
        [
            Joint.ELBOW_LEFT,
            Joint.WRIST_LEFT
        ],
        [
            Joint.WRIST_LEFT,
            Joint.HAND_LEFT
        ],
        [
            Joint.HAND_LEFT,
            Joint.HANDTIP_LEFT
        ],
        [
            Joint.WRIST_LEFT,
            Joint.THUMB_LEFT
        ],
        [
            Joint.NOSE,
            Joint.EYE_LEFT
        ],
        [
            Joint.EYE_LEFT,
            Joint.EAR_LEFT
        ],
        [
            Joint.SPINE_CHEST,
            Joint.CLAVICLE_RIGHT
        ],
        [
            Joint.CLAVICLE_RIGHT,
            Joint.SHOULDER_RIGHT
        ],
        [
            Joint.SHOULDER_RIGHT,
            Joint.ELBOW_RIGHT
        ],
        [
            Joint.ELBOW_RIGHT,
            Joint.WRIST_RIGHT
        ],
        [
            Joint.WRIST_RIGHT,
            Joint.HAND_RIGHT
        ],
        [
            Joint.HAND_RIGHT,
            Joint.HANDTIP_RIGHT
        ],
        [
            Joint.WRIST_RIGHT,
            Joint.THUMB_RIGHT
        ],
        [
            Joint.NOSE,
            Joint.EYE_RIGHT
        ],
        [
            Joint.EYE_RIGHT,
            Joint.EAR_RIGHT
        ]
]

parser = argparse.ArgumentParser()
parser.add_argument('--maxHands', nargs='?', default=6)
parser.add_argument('--minDetectionConfidence', nargs='?', default=0.6)
parser.add_argument('--minTrackingConfidence', nargs='?', default=0.6)
parser.add_argument('--videoPath', nargs='?', default="F:\\Weights_Task\\Data\\Fib_weights_original_videos\\")
parser.add_argument('--jsonPath', nargs='?', default="F:\\Weights_Task\\Data\\")
parser.add_argument('--fileName', nargs='?', default="Group_03-sub1")
parser.add_argument('--initialFrame', nargs='?', default=7000)

args = parser.parse_args()

# file
cap = cv2.VideoCapture("{}{}.mkv".format(args.videoPath, args.fileName))
cap.set(cv2.CAP_PROP_POS_FRAMES, args.initialFrame)
jsonFile = open("{}{}.json".format(args.jsonPath, args.fileName))

skeletonData = json.load(jsonFile)
frameData = skeletonData["frames"]
cameraCalibration = skeletonData["camera_calibration"]

#BGR
# Red, Green, Orange, Blue, Purple
colors = [(0, 0, 255), (0, 255, 0), (0, 140, 255), (255, 0, 0), (139,34,104)]
dotColor = [(0, 0, 139), (20,128,48), (71,130,170), (205,95,58), (205,150,205)]

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
shift = 7

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands(max_num_hands=args.maxHands, min_detection_confidence=args.minDetectionConfidence, min_tracking_confidence=args.minTrackingConfidence)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frameCount+=1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result_hands = hands.process(framergb)
    if result_hands.multi_hand_landmarks:
        landmarks = []
        for index, handslms in enumerate(result_hands.multi_hand_landmarks):
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * w)
                lmy = int(lm.y * h)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

    # if cameraCalibration != None:
    #     bodies = frameData[args.initialFrame + frameCount]["bodies"]
    #     for body in bodies:
    #         dictionary = {}
    #         for jointIndex, joint in enumerate(body["joint_positions"]):
    #             bodyLocation = getPointSubcategory(Joint(jointIndex))
    #             bodyId = int(body["body_id"])
    #             print(f"{bodyId}")
    #             if(bodyLocation != BodyCategory.RIGHT_LEG and bodyLocation != BodyCategory.LEFT_LEG):
    #                 points2D, _ = cv2.projectPoints(
    #                     np.array(joint), 
    #                     rotation,
    #                     translation,
    #                     cameraMatrix,
    #                     dist)  
                    
    #                 point = (int(points2D[0][0][0] * 2**shift),int(points2D[0][0][1] * 2**shift))
    #                 dictionary[Joint(jointIndex)] = point
    #                 cv2.circle(frame, point, radius=15, color=dotColor[bodyId % len(dotColor)], thickness=15, shift=shift)
    #         for bone in bone_list:
    #              cv2.line(frame, dictionary[bone[0]], dictionary[bone[1]], color=colors[bodyId % len(colors)], thickness=3, shift=shift)
                 

    cv2.putText(frame, "Frame: " + str(frameCount + args.initialFrame), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    frame = cv2.resize(frame, (960, 540))
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(10) == ord('q'):
        break
       
# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()