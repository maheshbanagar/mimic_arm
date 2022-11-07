import cv2
import mediapipe as mp
import numpy as np
import serial
import time

arduinoData = serial.Serial('com5',9600)

one=[]
two=[]
three=[]
count = 0
def calculate_angle(a,b,c):
    a = np.array(a) #first
    b = np.array(b) #mid
    c = np.array(c) #end

    radian = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radian*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mpHands = mp.solutions.hands
hands = mpHands.Hands()
fingerCoordinates = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumbCoordinate = (4,2)

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        #recolour image to rgb
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #make detection
        results = pose.process(image)
        resulthand = hands.process(image)

        #recolor back to bgr
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            multiLandMarks = resulthand.multi_hand_landmarks
            
            #get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
            thumb = [landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x,landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y]
            pinky = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x,landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
            
            #calculate angles
            angle1 = calculate_angle(hip,shoulder,elbow)
            angle2 = calculate_angle(shoulder,elbow,wrist)
            angle3 = calculate_angle(elbow,wrist,pinky)
            new_angle3 = abs((int(angle3)-150)*180/30 )

            cmd=str(int(angle1))
            cmd=cmd+'1\n'
            arduinoData.write(cmd.encode())
            cmd=str(int(angle2))
            cmd=cmd+'2\n'
            arduinoData.write(cmd.encode())
            cmd=str(int(new_angle3))
            cmd=cmd+'3\n'
            arduinoData.write(cmd.encode())
            print(new_angle3)
        
            #visualize angles
            cv2.putText(image, str(angle1),
                    tuple(np.multiply(shoulder, [640,480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, str(angle2),
                    tuple(np.multiply(elbow, [640,480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, str(angle3),
                    tuple(np.multiply(wrist, [640,480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

            
            handPoints = []
            for handLms in multiLandMarks:
                mp_drawing.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

                for idx, lm in enumerate(handLms.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    handPoints.append((cx, cy))

            for point in handPoints:
                cv2.circle(image, point, 10, (0, 0, 255), cv2.FILLED)

            upCount = 0
            for coordinate in fingerCoordinates:
                if handPoints[coordinate[0]][1] < handPoints[coordinate[1]][1]:
                    upCount += 1
            if handPoints[thumbCoordinate[0]][0] < handPoints[thumbCoordinate[1]][0]:
                upCount += 1
            cv2.putText(image, str(upCount), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (255,0,0), 12)
            cv2.putText(image,"open", (300,150), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 12)
            if upCount==0:  
                angle4=0
            else:
                angle4=170

    
        except:
            pass

        #render detections
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        cv2.imshow("Mediapipe Feed",image)    

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()