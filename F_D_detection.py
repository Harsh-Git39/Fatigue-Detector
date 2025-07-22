import cv2 as cv
import numpy as np
import pandas as pd 
import math
import mediapipe as mp
import winsound
# SET THRESHOLD VALUES OF EAR AND MAR
   
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#specific points1
LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144] 

RIGHT_EYE_POINTS = [362, 387, 385, 263, 373, 380]


MOUTH_POINTS = [78, 81, 13, 311, 308, 324, 318] 
 
#Euclidean distance for facial landmarks

def Euclidean_dist(point1 , point2):
    return math.sqrt((point2[0]-point1[0])**2 +(point2[1]-point1[1])**2)

#Eye aspect ratio
# eye_landmark is array of size 6 with each point as (x,y)

def Eye_Aspect_Ratio(eye_landmark):
    VERTICAL_DIST1 = Euclidean_dist(eye_landmark[1] , eye_landmark[5])
    VERTICAL_DIST2 = Euclidean_dist(eye_landmark[2] , eye_landmark[4])

    HORIZONTAL_DIST = Euclidean_dist(eye_landmark[0] , eye_landmark[3])

    EAR = (VERTICAL_DIST1 + VERTICAL_DIST2) / (2.0*HORIZONTAL_DIST)

    return EAR

#Mouth aspect ratio 

def Mouth_Aspect_Ratio(mouth_landmark):

    VERMOUTH1 = Euclidean_dist(mouth_landmark[2] , mouth_landmark[6])
    VERMOUTH2 = Euclidean_dist(mouth_landmark[4] , mouth_landmark[5])
    VERMOUTH3 = Euclidean_dist(mouth_landmark[3] , mouth_landmark[1])

    HORIMOUTH4 = Euclidean_dist(mouth_landmark[0] , mouth_landmark[4])

    MAR = (VERMOUTH1 + VERMOUTH2 + VERMOUTH3) / (3.0*HORIMOUTH4)

    return MAR

#Executive function

def faceDETECTOR(cam_opt):
    capture = cv.VideoCapture(cam_opt)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT , 480)
    capture.set(cv.CAP_PROP_FRAME_WIDTH  , 640)

    if not capture.isOpened():
        print("OOP's Machine cam not opened")
        return 
    
    print("Webcam opened successfully!")
    print("Press 'q' to quit")

    closed_eye_frames = 0
    open_mouth_frames = 0
    frame_count = 0

    
    with mp_face_mesh.FaceMesh(
         max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

      while True:
        #ret is boolean and frame is numpy array of image and read() returns tuple as takes two arguments
        ret , frame  = capture.read()
        frame_count += 1

        if not ret:
            print("Failed to read WEBCAM(0)")
            break

        frame = cv.flip(frame , 1)

        colorChange = cv.cvtColor(frame , cv.COLOR_BGR2RGB)

        process_Frame = face_mesh.process(colorChange)

        height , width = frame.shape[:2]

    #    frame = cv.cvtColor(frame , cv.COLOR_RGB2BGR)

        if process_Frame.multi_face_landmarks:
            for point in process_Frame.multi_face_landmarks:
                        
                landmarks = []

                for landmark in point.landmark:
      
                  x = int(landmark.x * width)
                  y = int(landmark.y * height)
 
                  landmarks.append([x,y])

                landmarks = np.array(landmarks)

            #for left and right eye will now store the specified landmark point in the landmarks list out of the 428 facial landmark point according to the mediapipe and set its own list 

                left_eye = [landmarks[i] for i in LEFT_EYE_POINTS]
                right_eye = [landmarks[i] for i in RIGHT_EYE_POINTS]

                mouth = [landmarks[i] for i in MOUTH_POINTS]

            #calc ear for both the eyes 
                ear_left_eye =  Eye_Aspect_Ratio(left_eye)
                ear_right_eye = Eye_Aspect_Ratio(right_eye)

            #avg ear for both the eyes
                avg_ear = (ear_left_eye + ear_right_eye)/2.0

                mar_mouth = Mouth_Aspect_Ratio(mouth)

                cv.putText(frame , f"EAR : {avg_ear:.4f}" , (10,30) ,cv.FONT_HERSHEY_SIMPLEX ,0.7 ,(0 , 255 , 0) , 2)
                cv.putText(frame , f"MAR : {mar_mouth:.4f}" , (10,60) ,cv.FONT_HERSHEY_SIMPLEX ,0.7 ,(0 , 255 , 0) , 2)


                if avg_ear < EAR_THRESHOLD:
                 closed_eye_frames+=1
                 cv.putText(frame, f"Closed frames: {closed_eye_frames}", (10, 90), 
                                  cv.FONT_HERSHEY_SIMPLEX, 0.65 , (255, 255, 0), 2)
                else:
                 closed_eye_frames = 0
                 cv.putText(frame, f"Yawn frames: {open_mouth_frames}", (10, 120), 
                                  cv.FONT_HERSHEY_SIMPLEX , 0.65 ,(255, 255, 0), 2)

                if mar_mouth > MAR_THRESHOLD:
                  open_mouth_frames+=1
                else:
                  open_mouth_frames = 0          


                if closed_eye_frames >= 15:
                  cv.putText(frame , "DROWSINESS ALERT!" , (200,120) , cv.FONT_HERSHEY_SIMPLEX , 1.2 ,( 255, 0 ,255) , 3)
                  cv.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 5)
                  winsound.Beep(700 , 600)
 

                if open_mouth_frames >= 10:
                  cv.putText(frame , "YAWNING ALERT!" ,(280,88) , cv.FONT_HERSHEY_SIMPLEX , 1.5 ,(255,  0 ,255) , 3)        
                  cv.rectangle(frame , (0,0) , (width,height),(255, 0, 255), 5) 
                  winsound.Beep(700 , 600)


        else:
                cv.putText(frame, "No face detected", (50, 50), 
                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow("ALERT" , frame)        

        if cv.waitKey(1) & 0XFF == ord('q'):   
            break 

    capture.release()
    cv.destroyAllWindows()

print("FOR WEBCAM : PRESS(0)")
print("FOR VIDEO FILE : PRESS(1)")

choose =input("ENTER YOUR CHOICE :")

if  choose =="0":
    faceDETECTOR(0)
elif choose =="1":
    video_path = input("Enter video file path: ")
    faceDETECTOR(video_path)
else:
     print("ENTER VALID OPTION!")       