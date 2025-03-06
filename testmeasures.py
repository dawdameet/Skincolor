import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

REFERENCE_HEIGHT_M = 1.7  # Default reference height
INCHES_PER_METER = 39.3701

def calculate_waist_width(landmarks, w, h):
    """Estimate waist width using anatomical proportions"""
    t = 0.6  # Waist position (60% from hip to shoulder)
    
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    # Calculate left and right waist points
    left_waist_x = left_hip.x + (left_shoulder.x - left_hip.x) * t
    left_waist_y = left_hip.y + (left_shoulder.y - left_hip.y) * t
    right_waist_x = right_hip.x + (right_shoulder.x - right_hip.x) * t
    right_waist_y = right_hip.y + (right_shoulder.y - right_hip.y) * t
    
    # Calculate waist width
    waist_width_px = math.hypot(
        (right_waist_x - left_waist_x) * w,
        (right_waist_y - left_waist_y) * h
    )
    
    return waist_width_px * math.pi  # Estimate circumference

def calculate_real_measurements(landmarks, image_shape):
    h, w = image_shape[:2]
    measurements = {}
    
    nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_heel = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
    right_heel = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
    
    # Calculate height
    heel_mid_y = (left_heel.y + right_heel.y) / 2
    height_px = max(abs(nose.y - heel_mid_y) * h, 1)  # Avoid division by zero
    pixels_to_meters = REFERENCE_HEIGHT_M / height_px
    
    # Shoulder width
    ls = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_px = math.hypot((rs.x - ls.x) * w, (rs.y - ls.y) * h)
    
    # Waist circumference
    lh = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    rh = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    
    if any(lm.visibility < 0.5 for lm in [ls, rs, lh, rh, nose, left_heel, right_heel]):
        raise ValueError("Key landmarks not visible")
    
    waist_circumference_px = calculate_waist_width(landmarks, w, h)
    
    # Convert to real-world units
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
            
            display_image = image.copy()
            measurement_text = ""

            if results.pose_landmarks:
                try:
                    measurements = calculate_real_measurements(results.pose_landmarks, image.shape)
                    mp_drawing.draw_landmarks(
                        display_image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2))
                    
                    measurement_text = (
                        f"Shoulder: {measurements['shoulder_in']:.1f}\"\n"
                        f"Waist: {measurements['waist_in']:.1f}\""
                    )
                except Exception as e:
                    measurement_text = "Error: Ensure full body is visible\nand stand facing the camera"

            # Draw measurement text
            y_pos = 30
            for line in measurement_text.split('\n'):
                cv2.putText(display_image, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                y_pos += 40

            # Flip the final annotated image
            display_image = cv2.flip(display_image, 1)

            cv2.imshow('Body Measurement System', display_image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_webcam()