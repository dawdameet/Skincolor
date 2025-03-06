import cv2
import mediapipe as mp
import numpy as np
import sys
import argparse

def extract_skin_tone(image: np.ndarray, exclude_beard: bool = False) -> np.ndarray:
    """
    Given an RGB image, detect the face using MediaPipe Face Mesh,
    create a smoothed convex hull mask for the face region using selected landmarks,
    optionally remove the lower portion (heuristically excluding beard),
    and compute the average skin tone weighted by brightness.
    """
    h, w, _ = image.shape

    # Initialize MediaPipe Face Mesh in static image mode
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,  # Process only one face per image
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image)
        if not results.multi_face_landmarks:
            print("No face detected in the image.")
            return None

        # Selected landmark indices focusing on key facial skin areas
        landmark_indices = [
            10, 67, 69, 104, 108, 109, 151,    # Forehead
            116, 117, 118, 119, 123, 147, 187,  # Upper cheeks
            168, 195, 197, 5, 4,               # Nose bridge
            50, 101, 100, 47, 205              # Eye sockets
        ]
        face_landmarks = results.multi_face_landmarks[0]
        # Get all available landmark points
        points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]
        # Append our selected landmarks to emphasize skin areas
        for i in landmark_indices:
            landmark = face_landmarks.landmark[i]
            points.append((int(landmark.x * w), int(landmark.y * h)))
        
        # Create a mask from the convex hull of all points
        mask = np.zeros((h, w), dtype=np.uint8)
        points_np = np.array(points, dtype=np.int32)
        cv2.fillConvexPoly(mask, points_np, 255)

        # Optionally exclude beard region
        if exclude_beard:
            # Heuristic: Use jawline indices (approximate indices) to remove lower face
            jawline_indices = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
            if len(jawline_indices) <= len(points):
                jaw_points = [points[i] for i in jawline_indices]
                y_min_jaw = min(p[1] for p in jaw_points)
                mask[y_min_jaw:, :] = 0

        # Use a combination of morphological closing and Gaussian blur to smooth the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # You might also experiment with a bilateral filter if needed:
        # mask_smoothed = cv2.bilateralFilter(mask_closed, d=9, sigmaColor=75, sigmaSpace=75)
        smooth_mask = cv2.GaussianBlur(mask_closed, (25, 25), 0)

        # Use the brightness (Value channel) to further refine the mask.
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        value_channel = hsv[:, :, 2]
        _, brightness_mask = cv2.threshold(value_channel, 40, 255, cv2.THRESH_BINARY)
        final_mask = cv2.bitwise_and(smooth_mask, brightness_mask)

        # Extract skin region and compute the average color weighted by brightness.
        skin_region = cv2.bitwise_and(image, image, mask=final_mask)
        indices = np.where(final_mask != 0)
        if len(indices[0]) == 0:
            return None
        weights = value_channel[indices].astype(float) / 255.0
        avg_color = np.average(skin_region[indices], axis=0, weights=weights)
        return avg_color.astype(np.uint8)

def process_images(image_paths: list, exclude_beard: bool) -> np.ndarray:
    skin_tones = []
    for path in image_paths:
        print(f"Processing {path}...")
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to load {path}")
            continue

        # Convert image from BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        avg_color = extract_skin_tone(rgb_img, exclude_beard=exclude_beard)
        if avg_color is not None:
            skin_tones.append(avg_color)
            print(f"Skin tone for {path}: {avg_color}")
        else:
            print(f"Could not extract skin tone from {path}")
    
    if not skin_tones:
        return None

    overall_avg = np.mean(skin_tones, axis=0).astype(np.uint8)
    return overall_avg

def main():
    parser = argparse.ArgumentParser(
        description="Extract overall average skin tone from images, optionally excluding beard regions."
    )
    parser.add_argument("images", nargs="+", help="Paths to input image files.")
    parser.add_argument("--exclude_beard", action="store_true",
                        help="Exclude beard region from skin tone calculation.")
    args = parser.parse_args()

    overall_skin_tone = process_images(args.images, exclude_beard=args.exclude_beard)
    if overall_skin_tone is not None:
        print("Final Overall Average Skin Tone (RGB):", overall_skin_tone.tolist())
        # Create a swatch for visualization
        swatch = np.zeros((100, 100, 3), dtype=np.uint8)
        swatch[:] = overall_skin_tone
        cv2.imshow("Overall Average Skin Tone", cv2.cvtColor(swatch, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No valid skin tone extracted from the images.")

if __name__ == "__main__":
    main()
