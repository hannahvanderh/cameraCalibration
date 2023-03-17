import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

def triangulate(mtx1, mtx2, R, T, frame):
 
    uvs1 = [[458, 86], [451, 164], [287, 181],
            [196, 383], [297, 444], [564, 194],
            [562, 375], [596, 520], [329, 620],
            [488, 622], [432, 52], [489, 56]]
 
    uvs2 = [[540, 311], [603, 359], [542, 378],
            [525, 507], [485, 542], [691, 352],
            [752, 488], [711, 605], [549, 651],
            [651, 663], [526, 293], [542, 290]]
 
    uvs1 = np.array(uvs1)
    uvs2 = np.array(uvs2)
 
    plt.imshow(frame[:,:,[2,1,0]])
    plt.scatter(uvs1[:,0], uvs1[:,1])
    plt.show() #this call will cause a crash if you use cv.imshow() above. Comment out cv.imshow() to see this.
 
    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = mtx1 @ RT1 #projection matrix for C1
 
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = mtx2 @ RT2 #projection matrix for C2
 
    def DLT(P1, P2, point1, point2):
 
        A = [point1[1]*P1[2,:] - P1[1,:],
             P1[0,:] - point1[0]*P1[2,:],
             point2[1]*P2[2,:] - P2[1,:],
             P2[0,:] - point2[0]*P2[2,:]
            ]
        A = np.array(A).reshape((4,4))
        #print('A: ')
        #print(A)
 
        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices = False)
 
        print('Triangulated point: ')
        print(Vh[3,0:3]/Vh[3,3])
        return Vh[3,0:3]/Vh[3,3]
 
    p3ds = []
    for uv1, uv2 in zip(uvs1, uvs2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)
 
    from mpl_toolkits.mplot3d import Axes3D
 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-15, 5)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(10, 30)
 
    connections = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [1,9], [2,8], [5,9], [8,9], [0, 10], [0, 11]]
    for _c in connections:
        print(p3ds[_c[0]])
        print(p3ds[_c[1]])
        ax.plot(xs = [p3ds[_c[0],0], p3ds[_c[1],0]], ys = [p3ds[_c[0],1], p3ds[_c[1],1]], zs = [p3ds[_c[0],2], p3ds[_c[1],2]], c = 'red')
    ax.set_title('This figure can be rotated.')
    #uncomment to see the triangulated pose. This may cause a crash if youre also using cv.imshow() above.
    plt.show()

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
h, w, c = frame.shape
frameCount = 0

#criteria used by checkerboard pattern detector.
#Change this if the code can't find the checkerboard
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
rows = 6 #number of checkerboard rows.
columns = 9 #number of checkerboard columns.
world_scaling = 1. #change this to the real world square size. Or not.
 
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
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frameCount+=1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if calibrated:
        bodies = frameData[args.initialFrame + frameCount]["bodies"]
        for body in bodies:
            for joint in body["joint_positions"]:
                points_2d, _ = cv2.projectPoints(np.array(joint), np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), cameraMatrix, dist)
                print(points_2d)
                print("width " + str(w) + " height " + str(h))
                cv2.circle(frame, (int(points_2d[0][0][0]),int(points_2d[0][0][1])), radius=5, color=(0, 0, 255), thickness=-1)

    if not calibrated:
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        GRID = (rows, columns)
        ret, corners = cv2.findChessboardCorners(gray, GRID, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            
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