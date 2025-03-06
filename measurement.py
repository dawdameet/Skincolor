import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

REFERENCE_HEIGHT_M = 1.5
INCHES_PER_METER = 39.3701

def calculate_real_measurements(landmarks, image_shape):
    h, w = image_shape[:2]
    measurements = {}
    
    nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_heel = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
    right_heel = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
    
    heel_mid_y = (left_heel.y + right_heel.y) / 2
    height_px = abs(nose.y - heel_mid_y) * h
    if height_px < 1: height_px = 1
    
    pixels_to_meters = REFERENCE_HEIGHT_M / height_px
    
    ls = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_px = math.hypot((rs.x - ls.x)*w, (rs.y - ls.y)*h)
    
    lh = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    rh = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    
    if lh.visibility < 0.5 or rh.visibility < 0.5:
        raise ValueError("Hip landmarks not visible")
    
    hip_width_px = math.hypot((rh.x - lh.x)*w, (rh.y - lh.y)*h)
    waist_circumference_px = hip_width_px * math.pi  
    
    measurements['shoulder_in'] = shoulder_px * pixels_to_meters * INCHES_PER_METER
    measurements['waist_in'] = waist_circumference_px * pixels_to_meters * INCHES_PER_METER
    
    return measurements

def process_webcam():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as pose:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success: continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            display_image = cv2.flip(image, 1)
            measurement_text = ""

            if results.pose_landmarks:
                try:
                    measurements = calculate_real_measurements(results.pose_landmarks, image.shape)
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2))
                    display_image = cv2.flip(image, 1)
                    measurement_text = f"Shoulder: {measurements['shoulder_in']:.1f}\"\nWaist: {measurements['waist_in']:.1f}\""
                except Exception as e:
                    measurement_text = "Stand facing camera\nwith hips visible"

            y_pos = 30
            for line in measurement_text.split('\n'):
                cv2.putText(display_image, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                y_pos += 40

            cv2.imshow('Body Measurement System', display_image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_webcam()