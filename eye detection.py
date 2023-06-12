
"""
Created on Fri Apr  8 09:40:16 2022

@author: amoolyagarg
"""

import cv2
import numpy as np
import dlib
import math
from datetime import datetime

now = datetime.now()
timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
file1 = open("MyFile.txt","a")
file1.write("\n")
file1.write(timestamp)
file1.write("\n")
file1.close()
cap = cv2.VideoCapture(0)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_pt = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_pt = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    hor_line = cv2.line(frame, left_pt, right_pt, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
    hor_line_ln = math.hypot((left_pt[0] - right_pt[0]), (left_pt[1] - right_pt[1]))
    ver_line_ln = math.hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_ln / ver_line_ln
    return ratio

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_gaze_ratio(eye_points,landmarks):
    left_eye_region = np.array([(landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y),
                        (landmarks.part(eye_points[1]).x, landmarks.part(eye_points[1]).y),
                        (landmarks.part(eye_points[2]).x, landmarks.part(eye_points[2]).y),
                        (landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y),
                        (landmarks.part(eye_points[4]).x, landmarks.part(eye_points[4]).y),
                        (landmarks.part(eye_points[5]).x, landmarks.part(eye_points[5]).y)], np.int32)
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])
    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
    eye = cv2.resize(gray_eye, None, fx=5, fy=5)

    
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    gaze_ratio = 0
    if right_side_white != 0:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
       
        landmarks = predictor(gray, face)
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))
        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
        
        left_point1 = (landmarks.part(42).x, landmarks.part(42).y)
        right_point1 = (landmarks.part(45).x, landmarks.part(45).y)
        center_top1 = midpoint(landmarks.part(43), landmarks.part(44))
        center_bottom1 = midpoint(landmarks.part(47), landmarks.part(46))
        hor_line1 = cv2.line(frame, left_point1, right_point1, (0, 255, 0), 2)
        ver_line1 = cv2.line(frame, center_top1, center_bottom1, (0, 255, 0), 2)
        
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        font = cv2.FONT_HERSHEY_SIMPLEX
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))
            file1 = open("MyFile.txt","a")
            file1.write("BLINKING\n")
            file1.close()
            print("Blinking")
        
            
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
        
        if gaze_ratio <= 1:
            cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
            file1 = open("MyFile.txt","a")
            file1.write("LEFT\n")
            file1.close()
            print("LEFT")
        elif 1 < gaze_ratio < 1.7:
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
            file1 = open("MyFile.txt","a")
            file1.write("CENTER\n")
            file1.close()
            print("CENTER")
        else:
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            print("RIGHT")
            file1 = open("MyFile.txt","a")
            file1.write("RIGHT\n")
            file1.close()
        cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()