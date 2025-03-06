import cv2
import mediapipe as mp
import numpy as np

class BodyShapeClassifier:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.7
        )
        
        # Updated body shape criteria with adjusted ratios
        self.body_shape_criteria = {
            'Hourglass': {
                'shoulder_hip_ratio': (0.95, 1.05),
                'waist_hip_ratio': (0.65, 0.75)
            },
            'Pear': {
                'shoulder_hip_ratio': (0.75, 0.9),
                'waist_hip_ratio': (0.7, 0.85)
            },
            'Apple': {
                'shoulder_hip_ratio': (0.9, 1.1),
                'waist_hip_ratio': (0.85, 1.0)
            },
            'Rectangle': {
                'shoulder_hip_ratio': (0.95, 1.05),
                'waist_hip_ratio': (0.8, 0.9)
            },
            'Inverted Triangle': {
                'shoulder_hip_ratio': (1.15, 1.3),
                'waist_hip_ratio': (0.7, 0.85)
            }
        }

    def classify_body_shape(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return "Invalid image", None

        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return "No body detected", None

        if not self._is_frontal_pose(results.pose_landmarks):
            return "Frontal pose required", None

        measurements = self._get_accurate_measurements(results.pose_landmarks, image.shape)
        shape = self._determine_body_shape(measurements)
        
        annotated_image = self._draw_measurements(image.copy(), results.pose_landmarks, measurements)
        return shape, annotated_image

    def _is_frontal_pose(self, landmarks):
        """Improved frontal pose detection"""
        left_ear = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
        return abs(left_ear.x - right_ear.x) < 0.2

    def _get_accurate_measurements(self, landmarks, image_shape):
        h, w = image_shape[:2]
        landmark = landmarks.landmark
        
        # Get key points using correct indices
        LEFT_SHOULDER = self.mp_pose.PoseLandmark.LEFT_SHOULDER
        RIGHT_SHOULDER = self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        LEFT_HIP = self.mp_pose.PoseLandmark.LEFT_HIP
        RIGHT_HIP = self.mp_pose.PoseLandmark.RIGHT_HIP
        
        # Calculate direct widths
        shoulder_width = self._calculate_distance(landmark[LEFT_SHOULDER], landmark[RIGHT_SHOULDER], w, h)
        hip_width = self._calculate_distance(landmark[LEFT_HIP], landmark[RIGHT_HIP], w, h)
        
        # Calculate waist using midpoint between shoulders and hips
        waist_point = type('', (object,), {
            "x": (landmark[LEFT_SHOULDER].x + landmark[RIGHT_SHOULDER].x)/2,
            "y": (landmark[LEFT_SHOULDER].y + landmark[LEFT_HIP].y)/2
        })
        
        # Calculate waist width as distance from midpoint to hips
        waist_width = self._calculate_distance(waist_point, landmark[LEFT_HIP], w, h) * 2

        return {
            'shoulder_hip_ratio': shoulder_width / hip_width,
            'waist_hip_ratio': waist_width / hip_width
        }

    def _calculate_distance(self, point1, point2, img_w, img_h):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(
            ((point1.x - point2.x) * img_w)**2 +
            ((point1.y - point2.y) * img_h)**2
        )


    def _determine_body_shape(self, measurements):
        """Improved classification logic with weighted scoring"""
        ratios = measurements
        shape_scores = {
            'Hourglass': 0,
            'Pear': 0,
            'Apple': 0,
            'Rectangle': 0,
            'Inverted Triangle': 0
        }

        # Score each shape based on how many criteria are met
        for shape, criteria in self.body_shape_criteria.items():
            for ratio_name, (min_val, max_val) in criteria.items():
                if min_val <= ratios[ratio_name] <= max_val:
                    shape_scores[shape] += 1

        # Get top scoring shapes
        max_score = max(shape_scores.values())
        candidates = [shape for shape, score in shape_scores.items() if score == max_score]

        # Tiebreaker rules
        if len(candidates) > 1:
            if 'Hourglass' in candidates and ratios['waist_hip_ratio'] < 0.75:
                return 'Hourglass'
            if 'Inverted Triangle' in candidates and ratios['shoulder_hip_ratio'] > 1.2:
                return 'Inverted Triangle'
            return candidates[0]
        
        return candidates[0] if max_score > 0 else 'Unknown'
    def _draw_measurements(self, image, landmarks, measurements):
        h, w = image.shape[:2]
        
        # Draw measurement lines
        self._draw_line(image, 
                       landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                       landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                       w, h, (0,255,255), 2)
        
        self._draw_line(image, 
                       landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP],
                       landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP],
                       w, h, (0,255,255), 2)

        # Add text annotations
        text = [
            f"Shoulder/Hip Ratio: {measurements['shoulder_hip_ratio']:.2f}",
            f"Waist/Hip Ratio: {measurements['waist_hip_ratio']:.2f}",
            f"Body Shape: {self._determine_body_shape(measurements)}"
        ]
        
        for i, line in enumerate(text):
            cv2.putText(image, line, (10, 30 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        return image

    def _draw_line(self, image, p1, p2, img_w, img_h, color, thickness):
        cv2.line(image,
                (int(p1.x * img_w), int(p1.y * img_h)),
                (int(p2.x * img_w), int(p2.y * img_h)),
                color, thickness)

if __name__ == "__main__":
    classifier = BodyShapeClassifier()
    image_path = "body2.png"
    
    shape, annotated_image = classifier.classify_body_shape(image_path)
    print(f"Body Shape: {shape}")
    
    if annotated_image is not None:
        cv2.imshow("Analysis", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()