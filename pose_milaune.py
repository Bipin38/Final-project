#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt


# In[2]:


import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose








def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 





def feedback_shavasana():
    
    cap = cv2.VideoCapture(0)
## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image,(600,400))
            # Extract landmarks
            
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, hip, knee)
            if 160 <= angle<= 180:
            # Visualize angle
                 mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                  
                                          
                                        )
                
            else:
                 mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2) 
                                     )   

            # Render detections
                       

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()








def feedback_padmasan():
    cap = cv2.VideoCapture(0)
## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image,(600,400))
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                

            
                left_shoulder_angle = calculate_angle(left_elbow,left_shoulder,left_hip)
                right_shoulder_angle = calculate_angle(right_elbow,right_shoulder,right_hip)
                left_hip_angle = calculate_angle(left_shoulder, left_hip , left_knee)
                right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                left_knee_angle = calculate_angle(left_hip,left_knee,left_ankle)
                right_knee_angle = calculate_angle(right_hip,right_knee,right_ankle)
            

                if 90 <= right_hip_angle <= 120 and 90<=left_hip_angle:
                    if 15<=left_knee_angle<= 40 and 15<= right_knee_angle <= 40:
                        if 10<=left_shoulder_angle<=30 and 10<=right_shoulder_angle<=30:
                            print("Perfect")
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) )
                            
                        else:
                            print("Bring you hand to the side of your leg.")
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))
                            
                    else: 
                        print("Fold you legs.")
                        
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))
                        
                
                    
                else :
                    print("Hip is not straight. Please make it straight.")
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))
                    
            except:
                pass
            
#             if 165 <= left_knee_angle <= 185 or 165 <= right_knee_angle <= 185:
           
                     
               
            
                       

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()




def feedback_vrikasana():
    cap = cv2.VideoCapture(0)
## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image,(600,400))
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                

                # Get coordinates
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                # Calculate angle
                

            
                left_shoulder_angle = calculate_angle(left_elbow,left_shoulder,left_hip)
                right_shoulder_angle = calculate_angle(right_elbow,right_shoulder,right_hip)
                left_hip_angle = calculate_angle(left_shoulder, left_hip , left_knee)
                right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                left_knee_angle = calculate_angle(left_hip,left_knee,left_ankle)
                right_knee_angle = calculate_angle(right_hip,right_knee,right_ankle)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow,right_wrist)
                left_elbow_angle = calculate_angle(left_shoulder,left_elbow,left_wrist)
    #             print(f'left_hip_angle {left_hip_angle}')
    #             print(f'right_hip_angle {right_hip_angle}')
    #             print(f'knee left {left_knee_angle}')
    #             print(f'right_knee_angle {right_knee_angle}')
    #             print(f'right_shoulder_angle {right_shoulder_angle}')
    #             print(f'left_shoulder_angle {left_shoulder_angle}')
    #             print(f'right_elbow_angle {right_elbow_angle}')
    #             print(f'left_elbow_angle {left_elbow_angle}')
                
                
                
    #             if 165 <= left_knee_angle <= 185 or 165 <= right_knee_angle <= 185:
                if 165 <= right_knee_angle <= 185:
                    if 30<=left_knee_angle<= 70:
                        if  130<=left_shoulder_angle<=180 and 130<=right_shoulder_angle<=180:
                        
                            if 130<=left_elbow_angle<=180 and 130<=right_elbow_angle<=180:
                    # Visualize angle
                                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 


                                                        )
                            else:
                                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2) 
                                        )
                        else:
                            
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2) 
                                        )
                                    
                    
                    else:
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2) 
                                        )   

                # Render detections
                else:
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2) 
                                        )           
            except:
                pass
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()





def feedback_bhuj():
    cap = cv2.VideoCapture(0)
## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image,(600,400))
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                # Calculate angle
                

            
                left_shoulder_angle = calculate_angle(left_elbow,left_shoulder,left_hip)
                right_shoulder_angle = calculate_angle(right_elbow,right_shoulder,right_hip)
                left_hip_angle = calculate_angle(left_shoulder, left_hip , left_knee)
                right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                left_knee_angle = calculate_angle(left_hip,left_knee,left_ankle)
                right_knee_angle = calculate_angle(right_hip,right_knee,right_ankle)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow,right_wrist)
                left_elbow_angle = calculate_angle(left_shoulder,left_elbow,left_wrist)
                # print(f'left_hip_angle {left_hip_angle}')
                # print(f'right_hip_angle {right_hip_angle}')
                # print(f'knee left {left_knee_angle}')
                # print(f'right_knee_angle {right_knee_angle}')
                # print(f'right_elbow_angle {right_elbow_angle}')
                
                
                
                
    #             if 165 <= left_knee_angle <= 185 or 165 <= right_knee_angle <= 185:
                if 150 <= right_knee_angle <= 170:
                    if 130<=right_elbow_angle<= 160:
                    
                        if  130<=right_hip_angle<=160:
                        
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) )
                        else:
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))
                            
                    else: 
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))
                
                    
                else :
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))
                    
                
                
    #             if 165 <= left_knee_angle <= 185 or 165 <= right_knee_angle <= 185:
            
                        
                    
                       
            except:
                pass
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()





def feedback_virbhadra():
    cap = cv2.VideoCapture(0)
## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image,(600,400))
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                # Calculate angle
                

                right_elbow_angle = calculate_angle(right_shoulder, right_elbow,right_wrist)
                left_elbow_angle = calculate_angle(left_shoulder,left_elbow,left_wrist)
                left_shoulder_angle = calculate_angle(left_elbow,left_shoulder,left_hip)
                right_shoulder_angle = calculate_angle(right_elbow,right_shoulder,right_hip)
                left_hip_angle = calculate_angle(left_shoulder, left_hip , left_knee)
                right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                left_knee_angle = calculate_angle(left_hip,left_knee,left_ankle)
                right_knee_angle = calculate_angle(right_hip,right_knee,right_ankle)
                
                # print(f'left_hip_angle {left_hip_angle}')
                # print(f'right_hip_angle {right_hip_angle}')
                # print(f'knee left {left_knee_angle}')
                # print(f'right_knee_angle {right_knee_angle}')
                # print(f'right_shoulder_angle {right_shoulder_angle}')
                # print(f'left_shoulder_angle {left_shoulder_angle}')
                
                
                
                
    #             if 165 <= left_knee_angle <= 185 or 165 <= right_knee_angle <= 185:
                if 170 <= left_knee_angle <= 185:
                    if 90<=right_knee_angle<= 110:
                    
                        if  60<=right_shoulder_angle<=100 and 60<=left_shoulder_angle<=100:
                        
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) )
                        else:
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))
                            
                    else: 
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))
                
                    
                else :
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))
                    
                
                
    #             if 165 <= left_knee_angle <= 185 or 165 <= right_knee_angle <= 185:
            
                        
                    
                       
            except:
                pass
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()





def feedback_tadasana():
    cap = cv2.VideoCapture(0)
## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image,(600,400))
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                # Calculate angle
                

                right_elbow_angle = calculate_angle(right_shoulder, right_elbow,right_wrist)
                left_elbow_angle = calculate_angle(left_shoulder,left_elbow,left_wrist)
                left_shoulder_angle = calculate_angle(left_elbow,left_shoulder,right_shoulder)
                right_shoulder_angle = calculate_angle(right_elbow,right_shoulder,left_shoulder)
                left_hip_angle = calculate_angle(left_shoulder, left_hip , left_knee)
                right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                left_knee_angle = calculate_angle(left_hip,left_knee,left_ankle)
                right_knee_angle = calculate_angle(right_hip,right_knee,right_ankle)
                
                # print(f'left_hip_angle {left_hip_angle}')
                # print(f'right_hip_angle {right_hip_angle}')
                # print(f'knee left {left_knee_angle}')
                # print(f'right_knee_angle {right_knee_angle}')
                # print(f'right_shoulder_angle {right_shoulder_angle}')
                # print(f'left_shoulder_angle {left_shoulder_angle}')
                
                
                
                
    #             if 165 <= left_knee_angle <= 185 or 165 <= right_knee_angle <= 185:
                if 160 <= right_hip_angle <= 185 and 160<=left_hip_angle<=185:
                    if 60<=right_shoulder_angle<= 95 and 60<= left_shoulder_angle<=95:
                    
                    
                        
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) )
                        
                    else: 
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))
                
                    
                else :
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))
                    
            except:
                pass    
            
#             if 165 <= left_knee_angle <= 185 or 165 <= right_knee_angle <= 185:
           
                     
                
                       

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()










