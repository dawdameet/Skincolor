import cv2
import mediapipe as mp
import numpy as np
import sys
import argparse
import csv

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
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image)
        if not results.multi_face_landmarks:
            print("No face detected in the image.")
            return None
        landmark_indices = [
            10, 67, 69, 104, 108, 109, 151,   # Forehead
            116, 117, 118, 119, 123, 147, 187, # Upper cheeks
            168, 195, 197, 5, 4,              # Nose bridge
            50, 101, 100, 47, 205             # Eye sockets
        ]
        face_landmarks = results.multi_face_landmarks[0]
        points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]
        for i in landmark_indices:
            landmark = face_landmarks.landmark[i]
            points.append((int(landmark.x * w), int(landmark.y * h)))
        mask = np.zeros((h, w), dtype=np.uint8)
        points_np = np.array(points, dtype=np.int32)
        cv2.fillConvexPoly(mask, points_np, 255)
        
        if exclude_beard:
            # Heuristic: Exclude jawline by masking out lower part of face bounding box
            jawline_indices = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
            jaw_points = [points[i] for i in jawline_indices if i < len(points)]
            if jaw_points:
                y_min_jaw = min(p[1] for p in jaw_points)
                mask[y_min_jaw:, :] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        smooth_mask = cv2.GaussianBlur(mask, (25, 25), 0)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        value_channel = hsv[:, :, 2]
        _, brightness_mask = cv2.threshold(value_channel, 40, 255, cv2.THRESH_BINARY)
        final_mask = cv2.bitwise_and(smooth_mask, brightness_mask)
        skin_region = cv2.bitwise_and(image, image, mask=final_mask)
        indices = np.where(final_mask != 0)
        if len(indices[0]) == 0:
            return None
        weights = value_channel[indices].astype(float) / 255.0
        avg_color = np.average(skin_region[indices], axis=0, weights=weights)
        return avg_color.astype(np.uint8)
    
def create_swatch(color: np.ndarray, size: int = 100) -> np.ndarray:
    """Create a square color swatch of the given RGB color."""
    swatch = np.zeros((size, size, 3), dtype=np.uint8)
    swatch[:] = color
    return swatch

def process_and_label(image_paths: list, exclude_beard: bool, output_csv: str):
    data = []
    for path in image_paths:
        print(f"\nProcessing {path}...")
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to load {path}")
            continue

        # Convert image from BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        extracted = extract_skin_tone(rgb_img, exclude_beard=exclude_beard)
        if extracted is None:
            print(f"Could not extract skin tone from {path}")
            continue

        print(f"Extracted skin tone (RGB): {extracted.tolist()}")
        swatch = create_swatch(extracted)
        cv2.imshow("Extracted Skin Tone", cv2.cvtColor(swatch, cv2.COLOR_RGB2BGR))
        # Show the original image for reference
        cv2.imshow("Original Image", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)  # small delay to update windows

        # Ask if the extracted color is correct
        confirm = input("Is the extracted color correct? (y/n): ").strip().lower()
        if confirm == "y":
            final_rgb = extracted
        else:
            # Prompt for custom input until a valid value is provided
            while True:
                user_input = input("Enter the correct skin tone as R,G,B: ").strip()
                try:
                    final_rgb = np.array([int(x) for x in user_input.split(",")], dtype=np.uint8)
                    if final_rgb.shape[0] != 3:
                        raise ValueError("You must enter 3 values.")
                    break
                except Exception as e:
                    print(f"Invalid input: {e}. Please try again.")
        cv2.destroyAllWindows()
        data.append([path, final_rgb[0], final_rgb[1], final_rgb[2]])
    
    if data:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "R", "G", "B"])
            writer.writerows(data)
        print(f"\nDataset saved to {output_csv}")
    else:
        print("No data collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactively build a skin tone dataset. For each image, the program extracts a skin tone, asks for confirmation, and if incorrect, lets you input the correct RGB value."
    )
    parser.add_argument("images", nargs="+", help="Paths to input image files.")
    parser.add_argument("--exclude_beard", action="store_true", help="Exclude beard region from extraction.")
    parser.add_argument("--output_csv", type=str, default="skin_tone_dataset.csv", help="Output CSV file for the dataset.")
    args = parser.parse_args()

    process_and_label(args.images, args.exclude_beard, args.output_csv)
