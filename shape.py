import cv2
import mediapipe as mp
import numpy as np

class BodyShapeClassifier:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.7
        )
        
        self.body_shape_criteria = {
            'Hourglass': {'shoulder_hip_ratio': (0.95, 1.05), 'waist_hip_ratio': (0.65, 0.75)},
            'Pear': {'shoulder_hip_ratio': (0.75, 0.89), 'waist_hip_ratio': (0.7, 0.85)},
            'Apple': {'shoulder_hip_ratio': (0.9, 1.1), 'waist_hip_ratio': (0.85, 1.0)},
            'Rectangle': {'shoulder_hip_ratio': (0.95, 1.05), 'waist_hip_ratio': (0.8, 0.95)},
            'Inverted Triangle': {'shoulder_hip_ratio': (1.1, 1.3), 'waist_hip_ratio': (0.7, 0.85)}
        }

    def classify_body_shape(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return "Invalid image", None

        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return "No body detected", None

        measurements = self._get_accurate_measurements(results.pose_landmarks, image.shape)
        
        if measurements['waist_hip_ratio'] == 0:
            return "Measurement error - try different photo", None
            
        shape = self._determine_body_shape(measurements)
        return shape, self._visualize_results(image.copy(), results.pose_landmarks, measurements)

    def _get_accurate_measurements(self, landmarks, img_shape):
        h, w = img_shape[:2]
        lms = landmarks.landmark
        
        # Get key landmarks
        LS = self.mp_pose.PoseLandmark.LEFT_SHOULDER
        RS = self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        LH = self.mp_pose.PoseLandmark.LEFT_HIP
        RH = self.mp_pose.PoseLandmark.RIGHT_HIP

        # Calculate direct measurements
        shoulder_width = self._distance(lms[LS], lms[RS], w, h)
        hip_width = self._distance(lms[LH], lms[RH], w, h)
        
        # Improved waist measurement using torso proportions
        waist_width = self._calculate_waist_width(lms, w, h, hip_width)
        
        return {
            'shoulder_hip_ratio': shoulder_width / hip_width,
            'waist_hip_ratio': waist_width / hip_width
        }

    def _calculate_waist_width(self, lms, img_w, img_h, hip_width):
        """Calculate waist width using multiple fallback methods"""
        try:
            # Method 1: Use shoulder-hip midpoint
            mid_torso_y = (lms[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y + 
                          lms[self.mp_pose.PoseLandmark.LEFT_HIP].y) / 2
            left_point = self._find_body_edge(lms, mid_torso_y, 'left', img_w, img_h)
            right_point = self._find_body_edge(lms, mid_torso_y, 'right', img_w, img_h)
            waist_width = abs(right_point - left_point)
            
            # Validate against hip width
            if waist_width <= hip_width * 0.3 or waist_width >= hip_width * 1.5:
                raise ValueError("Invalid waist measurement")
                
        except:
            # Fallback to anthropometric average
            waist_width = hip_width * 0.85
        
        return waist_width

    def _find_body_edge(self, lms, y_pos, side, img_w, img_h):
        """Find horizontal body edge at given vertical position"""
        hip = lms[self.mp_pose.PoseLandmark.LEFT_HIP if side == 'left' else self.mp_pose.PoseLandmark.RIGHT_HIP]
        shoulder = lms[self.mp_pose.PoseLandmark.LEFT_SHOULDER if side == 'left' else self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Calculate horizontal position using body proportions
        return hip.x * 0.7 + shoulder.x * 0.3

    def _determine_body_shape(self, measurements):
        ratios = measurements
        for shape, criteria in self.body_shape_criteria.items():
            if (criteria['shoulder_hip_ratio'][0] <= ratios['shoulder_hip_ratio'] <= criteria['shoulder_hip_ratio'][1] and
                criteria['waist_hip_ratio'][0] <= ratios['waist_hip_ratio'] <= criteria['waist_hip_ratio'][1]):
                return shape
        return 'Unknown'

    def _visualize_results(self, image, landmarks, measurements):
        h, w = image.shape[:2]
        
        # Draw measurement lines
        self._draw_line(image, landmarks, 'LEFT_SHOULDER', 'RIGHT_SHOULDER', w, h)
        self._draw_line(image, landmarks, 'LEFT_HIP', 'RIGHT_HIP', w, h)
        
        # Add ratio text
        text = [
            f"Shoulder/Hip Ratio: {measurements['shoulder_hip_ratio']:.2f}",
            f"Waist/Hip Ratio: {measurements['waist_hip_ratio']:.2f}",
            f"Body Shape: {self._determine_body_shape(measurements)}"
        ]
        
        for i, line in enumerate(text):
            cv2.putText(image, line, (20, 40 + i*40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
        return image

    def _draw_line(self, image, landmarks, point1, point2, img_w, img_h):
        p1 = getattr(self.mp_pose.PoseLandmark, point1)
        p2 = getattr(self.mp_pose.PoseLandmark, point2)
        cv2.line(image,
                (int(landmarks.landmark[p1].x * img_w), int(landmarks.landmark[p1].y * img_h)),
                (int(landmarks.landmark[p2].x * img_w), int(landmarks.landmark[p2].y * img_h)),
                (0,255,255), 2)

    def _distance(self, p1, p2, w, h):
        return np.hypot((p1.x - p2.x)*w, (p1.y - p2.y)*h)

if __name__ == "__main__":
    analyzer = BodyShapeClassifier()
    image_path = "body3.png"
    
    shape, result_img = analyzer.classify_body_shape(image_path)
    print(f"Body Shape: {shape}")
    
    if result_img is not None:
        cv2.imshow("Body Shape Analysis", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()