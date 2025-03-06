import cv2
import mediapipe as mp
import numpy as np
import sys

def extract_skin_tone(image):
    """
    Given an RGB image, detect the face using MediaPipe Face Mesh,
    create a smoothed convex hull mask for the face region, and compute the average skin tone.
    """
    h, w, _ = image.shape

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1, 
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image)
        if not results.multi_face_landmarks:
            print("No face detected in the image.")
            return None

        face_landmarks = results.multi_face_landmarks[0]
        points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]

        mask = np.zeros((h, w), dtype=np.uint8)
        points_np = np.array(points, dtype=np.int32)
        cv2.fillConvexPoly(mask, points_np, 255)

        smooth_mask = cv2.GaussianBlur(mask, (15, 15), 0)

        skin_region = cv2.bitwise_and(image, image, mask=smooth_mask)
        
        indices = np.where(smooth_mask != 0)
        if len(indices[0]) == 0:
            print("No skin region detected in the image.")
            return None
        
        avg_color = np.mean(skin_region[indices], axis=0)
        return avg_color.astype(np.uint8)

def process_images(image_paths):
    skin_tones = []
    for path in image_paths:
        print(f"Processing {path}...")
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to load {path}")
            continue

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        avg_color = extract_skin_tone(rgb_img)
        if avg_color is not None:
            skin_tones.append(avg_color)
            print(f"Skin tone for {path}: {avg_color}")
        else:
            print(f"Could not extract skin tone from {path}")
    
    if not skin_tones:
        return None
    overall_avg = np.mean(skin_tones, axis=0).astype(np.uint8)
    return overall_avg

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python final_skin_tone.py <image_path1> <image_path2> ...")
        sys.exit(1)
    
    image_paths = sys.argv[1:]
    overall_skin_tone = process_images(image_paths)
    if overall_skin_tone is not None:
        print("Final Overall Average Skin Tone (RGB):", overall_skin_tone.tolist())
        
        swatch = np.zeros((100, 100, 3), dtype=np.uint8)
        swatch[:] = overall_skin_tone
        cv2.imshow("Overall Average Skin Tone", cv2.cvtColor(swatch, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No valid skin tone extracted from the images.")
