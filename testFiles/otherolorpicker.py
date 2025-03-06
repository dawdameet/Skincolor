#!/usr/bin/env python3
"""
Enhanced Skin Tone Extractor using MediaPipe

This script detects faces in images, extracts the skin region using a convex hull
of facial landmarks, smooths the mask edges, and computes statistical color measures
in multiple color spaces. It supports parallel processing, multiple faces per image,
visualizations, and flexible output (JSON/CSV).

Usage examples:
    python enhanced_skin_tone_extractor.py --images img1.jpg img2.jpg --all_faces --output_format json
    python enhanced_skin_tone_extractor.py --images img1.jpg --save_vis --vis_dir visualizations
"""

import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import argparse
import json
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Any
from statistics import median

# Define helper functions for color space conversion
def convert_to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0][0]

def convert_to_hsv(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]

def compute_color_statistics(region: np.ndarray) -> Dict[str, Any]:
    """Compute mean, median, 10th and 90th percentile for each channel (in RGB)."""
    # region: array of shape (num_pixels, 3)
    stats = {}
    for i, channel in enumerate(['R', 'G', 'B']):
        data = region[:, i]
        stats[f"{channel}_mean"] = float(np.mean(data))
        stats[f"{channel}_median"] = float(median(data))
        stats[f"{channel}_p10"] = float(np.percentile(data, 10))
        stats[f"{channel}_p90"] = float(np.percentile(data, 90))
    return stats

def create_color_swatch(color: np.ndarray, size: int = 100) -> np.ndarray:
    """Create an image filled with the provided color (RGB)."""
    swatch = np.zeros((size, size, 3), dtype=np.uint8)
    swatch[:] = color
    return swatch

def smooth_mask(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Apply a Gaussian blur to the mask edges to create a smooth transition."""
    return cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

def process_single_image(image_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Process one image: detect faces, extract skin regions,
    compute color statistics in RGB, LAB, and HSV.
    Returns a dictionary of results.
    """
    result = {"image_path": image_path, "faces": []}
    if not os.path.exists(image_path):
        result["error"] = "File does not exist"
        return result

    img = cv2.imread(image_path)
    if img is None:
        result["error"] = "Failed to load image"
        return result

    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb_img.shape

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,  # allow up to 5 faces
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        mp_results = face_mesh.process(rgb_img)
        if not mp_results.multi_face_landmarks:
            result["error"] = "No face detected"
            return result

        # Decide which faces to process:
        faces_to_process = mp_results.multi_face_landmarks
        if not args.all_faces:
            # Process only the largest face based on bounding box area
            areas = []
            for face in mp_results.multi_face_landmarks:
                pts = np.array([(int(landmark.x * w), int(landmark.y * h)) for landmark in face.landmark])
                x, y, w_rect, h_rect = cv2.boundingRect(pts.astype(np.int32))
                areas.append(w_rect * h_rect)
            largest_idx = int(np.argmax(areas))
            faces_to_process = [mp_results.multi_face_landmarks[largest_idx]]

        for face_landmarks in faces_to_process:
            # Get all landmark points
            pts = np.array([(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark])
            # Create convex hull mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts, 255)
            # Smooth the mask edges
            smooth = smooth_mask(mask, kernel_size=args.blur_kernel)
            # Extract skin region using the smooth mask (mask as weight)
            # First, convert mask to float and normalize
            mask_norm = smooth.astype(np.float32) / 255.0
            # Multiply the RGB image by the mask (expand mask to 3 channels)
            skin_region = rgb_img * mask_norm[:, :, None]
            # Get only the pixels within the mask
            indices = np.where(smooth > 0)
            if len(indices[0]) == 0:
                continue
            skin_pixels = skin_region[indices]
            # Compute statistics in RGB
            rgb_stats = compute_color_statistics(skin_pixels)
            # Also compute overall average in RGB
            avg_rgb = np.mean(skin_pixels, axis=0).astype(np.uint8)
            # Convert average color to LAB and HSV
            lab_color = convert_to_lab(avg_rgb).tolist()
            hsv_color = convert_to_hsv(avg_rgb).tolist()

            face_result = {
                "bounding_box": cv2.boundingRect(pts.astype(np.int32)),
                "rgb_stats": rgb_stats,
                "avg_rgb": avg_rgb.tolist(),
                "avg_lab": lab_color,
                "avg_hsv": hsv_color,
            }

            # If visualization is enabled, create a composite image
            if args.save_vis:
                # Create a color swatch image for average color in RGB
                swatch = create_color_swatch(avg_rgb, size=100)
                # Create a visualization: original, mask, extracted skin, and swatch side by side
                mask_vis = cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR)
                extracted_vis = cv2.bitwise_and(rgb_img, rgb_img, mask=smooth)
                # Resize images for display
                disp_orig = cv2.resize(rgb_img, (200, 200))
                disp_mask = cv2.resize(mask_vis, (200, 200))
                disp_extracted = cv2.resize(extracted_vis, (200, 200))
                disp_swatch = cv2.resize(swatch, (200, 200))
                # Concatenate horizontally
                composite = np.hstack((disp_orig, disp_mask, disp_extracted, disp_swatch))
                # Convert composite from RGB to BGR for saving via OpenCV
                composite_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                vis_path = os.path.join(args.vis_dir, f"{base_name}_face_{len(result['faces'])}.jpg")
                os.makedirs(args.vis_dir, exist_ok=True)
                cv2.imwrite(vis_path, composite_bgr)
                face_result["vis_path"] = vis_path

            result["faces"].append(face_result)

    return result

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Skin Tone Extractor using MediaPipe and Color Analysis."
    )
    parser.add_argument("--images", nargs="+", required=True, help="Paths to input images.")
    parser.add_argument("--all_faces", action="store_true", help="Process all detected faces; otherwise, only the largest face is used.")
    parser.add_argument("--blur_kernel", type=int, default=15, help="Kernel size for Gaussian blur on mask edges (must be odd).")
    parser.add_argument("--output_format", choices=["json", "csv"], default="json", help="Output format for results.")
    parser.add_argument("--output_file", type=str, default="results", help="Output file base name (without extension).")
    parser.add_argument("--save_vis", action="store_true", help="Save visualization images for each processed face.")
    parser.add_argument("--vis_dir", type=str, default="visualizations", help="Directory to save visualizations.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers for processing images.")
    args = parser.parse_args()

    # Process images in parallel if more than 1 worker is specified
    results = []
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_single_image, img_path, args): img_path for img_path in args.images}
            for future in as_completed(futures):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    print(f"Error processing {futures[future]}: {e}")
    else:
        for img_path in args.images:
            res = process_single_image(img_path, args)
            results.append(res)

    # Save results in the desired output format
    if args.output_format == "json":
        out_file = f"{args.output_file}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {out_file}")
    else:
        out_file = f"{args.output_file}.csv"
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["image_path", "face_index", "bounding_box", "avg_rgb", "avg_lab", "avg_hsv", "rgb_stats", "vis_path"]
            writer.writerow(header)
            for res in results:
                image_path = res.get("image_path", "")
                for idx, face in enumerate(res.get("faces", [])):
                    writer.writerow([
                        image_path,
                        idx,
                        face.get("bounding_box", ""),
                        face.get("avg_rgb", ""),
                        face.get("avg_lab", ""),
                        face.get("avg_hsv", ""),
                        json.dumps(face.get("rgb_stats", {})),
                        face.get("vis_path", "")
                    ])
        print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
