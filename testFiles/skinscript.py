import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import json
import logging
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SkinToneExtractor:
    def __init__(self, 
                 max_faces: int = 1,  # Set default to 1
                 mask_blur_kernel: int = 21,
                 color_spaces: List[str] = ['rgb', 'lab', 'hsv'],
                 output_dir: str = 'results'):
        if max_faces < 1:
            raise ValueError("max_faces should be greater than or equal to 1")
        self.max_faces = max_faces
        self.mask_blur_kernel = mask_blur_kernel
        self.color_spaces = [cs.lower() for cs in color_spaces]
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.valid_color_spaces = {'rgb', 'lab', 'hsv', 'ycrcb'}
        invalid = set(self.color_spaces) - self.valid_color_spaces
        if invalid:
            raise ValueError(f"Invalid color spaces: {invalid}")

    def process_image_batch(self, image_paths: List[str]) -> Dict:
        """Process multiple images with multiprocessing"""
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(self._process_single_image, image_paths)
        
        # Aggregate results
        aggregated = defaultdict(list)
        for result in results:
            if not result:
                continue
            for key, value in result.items():
                aggregated[key].append(value)
        
        # Calculate overall averages
        final_results = {}
        for cs in self.color_spaces:
            if f'{cs}_averages' in aggregated:
                avg = np.mean([a['average'] for a in aggregated[f'{cs}_averages']], axis=0)
                final_results[f'overall_{cs}_average'] = avg.tolist()
        
        return final_results

    def _process_single_image(self, image_path: str) -> Optional[Dict]:
        """Process a single image and return skin tone analysis"""
        try:
            with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=self.max_faces,
                refine_landmarks=True,
                min_detection_confidence=0.3
            ) as face_mesh:
                image = cv2.imread(image_path)
                if image is None:
                    logging.warning(f"Invalid image: {image_path}")
                    return None

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_image)
                
                if not results.multi_face_landmarks:
                    logging.warning(f"No faces detected in {image_path}")
                    return None

                face_results = []
                for face_landmarks in results.multi_face_landmarks:
                    face_result = self._process_face(rgb_image, face_landmarks)
                    if face_result:
                        face_results.append(face_result)

                if not face_results:
                    return None

                # Save visualization
                self._save_visualization(image_path, face_results)

                return {
                    'image': image_path,
                    'faces_processed': len(face_results),
                    **{k: v for res in face_results for k, v in res.items()}
                }

        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            return None

    def _process_face(self, image: np.ndarray, landmarks) -> Optional[Dict]:
        """Process individual face and extract skin tone information"""
        h, w, _ = image.shape
        points = np.array([(lm.x * w, lm.y * h) for lm in landmarks.landmark])  
        mask = np.zeros((h, w), dtype=np.uint8)
        convex_hull = cv2.convexHull(points.astype(np.int32))
        cv2.fillConvexPoly(mask, convex_hull, 255)
        mask = cv2.GaussianBlur(mask, (self.mask_blur_kernel, self.mask_blur_kernel), 0)
        skin_region = cv2.bitwise_and(image, image, mask=mask)
        non_zero_pixels = skin_region[mask != 0]
        if len(non_zero_pixels) == 0:
            return None
        results = {}
        for cs in self.color_spaces:
            converted = self._convert_color_space(non_zero_pixels, cs)
            avg = np.mean(converted, axis=0)
            med = np.median(converted, axis=0)
            std = np.std(converted, axis=0)
            results.update({
                f'{cs}_average': avg.tolist(),
                f'{cs}_median': med.tolist(),
                f'{cs}_std_dev': std.tolist(),
                f'{cs}_percentile_25': np.percentile(converted, 25, axis=0).tolist(),
                f'{cs}_percentile_75': np.percentile(converted, 75, axis=0).tolist()
            })
        
        return results

    def _convert_color_space(self, pixels: np.ndarray, color_space: str) -> np.ndarray:
        """Convert pixel values to specified color space"""
        if color_space == 'rgb':
            return pixels
        elif color_space == 'lab':
            return cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
        elif color_space == 'hsv':
            return cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        elif color_space == 'ycrcb':
            return cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2YCrCb).reshape(-1, 3)
        else:
            raise ValueError(f"Unsupported color space: {color_space}")

    def _save_visualization(self, image_path: str, face_results: List[Dict]):
        """Create and save visualization images"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(self.output_dir, f"{base_name}_analysis.png")
        
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0,0].imshow(rgb_image)
        axes[0,0].set_title('Original Image')
        axes[0,0].axis('off')
        
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            face_results[0]['landmarks'],
            mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        )
        axes[0,1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0,1].set_title('Face Mesh')
        axes[0,1].axis('off')
        
        mask = np.zeros_like(image[:, :, 0])
        for res in face_results:
            h, w, _ = image.shape
            points = np.array([(lm.x * w, lm.y * h) for lm in res['landmarks'].landmark])
            convex_hull = cv2.convexHull(points.astype(np.int32))
            cv2.fillConvexPoly(mask, convex_hull, 255)
        mask = cv2.GaussianBlur(mask, (self.mask_blur_kernel, self.mask_blur_kernel), 0)
        axes[1,0].imshow(mask, cmap='gray')
        axes[1,0].set_title('Skin Mask')
        axes[1,0].axis('off')
        
        swatch_size = 100
        swatches = np.zeros((swatch_size*len(self.color_spaces), swatch_size, 3), dtype=np.uint8)
        for i, cs in enumerate(self.color_spaces):
            avg_color = np.array(face_results[0][f'{cs}_average'])
            if cs != 'rgb':
                color_patch = np.full((1, 1, 3), avg_color, dtype=np.uint8)
                if cs == 'lab':
                    color_patch = cv2.cvtColor(color_patch, cv2.COLOR_LAB2RGB)
                elif cs == 'hsv':
                    color_patch = cv2.cvtColor(color_patch, cv2.COLOR_HSV2RGB)
                elif cs == 'ycrcb':
                    color_patch = cv2.cvtColor(color_patch, cv2.COLOR_YCrCb2RGB)
                avg_color = color_patch[0][0]
            swatches[i*swatch_size:(i+1)*swatch_size, :, :] = avg_color
        axes[1,1].imshow(swatches)
        axes[1,1].set_title('Average Colors')
        axes[1,1].axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    def save_results(self, results: Dict, format: str = 'json'):
        """Save aggregated results in specified format"""
        if format == 'json':
            with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
                json.dump(results, f, indent=4)
        elif format == 'csv':
            pass
        else:
            raise ValueError(f"Unsupported format: {format}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Skin Tone Extraction Tool")
    parser.add_argument('images', nargs='+', help="Input image paths")
    parser.add_argument('--max-faces', type=int, default=1,
                       help="Maximum number of faces to detect (must be >= 1)")
    parser.add_argument('--blur-kernel', type=int, default=21, 
                       help="Mask blur kernel size (odd number)")
    parser.add_argument('--color-spaces', nargs='+', default=['rgb', 'lab'],
                       help="Color spaces to analyze (rgb, lab, hsv, ycrcb)")
    parser.add_argument('--output-dir', default='results', 
                       help="Output directory for results")
    parser.add_argument('--format', choices=['json', 'csv'], default='json',
                       help="Output format")

    args = parser.parse_args()

    extractor = SkinToneExtractor(
        max_faces=args.max_faces,
        mask_blur_kernel=args.blur_kernel,
        color_spaces=args.color_spaces,
        output_dir=args.output_dir
    )

    results = extractor.process_image_batch(args.images)
    extractor.save_results(results, format=args.format)