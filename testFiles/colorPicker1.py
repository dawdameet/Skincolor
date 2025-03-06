import cv2
import mediapipe as mp
import numpy as np
import sys
import argparse
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('skin_tone.log'), logging.StreamHandler()]
)

class SkinToneAnalyzer:
    def __init__(self, num_processes: int = None):
        self.num_processes = num_processes or cpu_count()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def _create_face_mask(self, image: np.ndarray, landmarks) -> np.ndarray:
        """Create a refined face mask using facial landmarks"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        face_oval = mp.solutions.face_mesh.FACEMESH_FACE_OVAL
        connections = [tuple(connection) for connection in np.array(face_oval).tolist()]
        
        points = []
        for connection in connections:
            points.extend([landmarks[connection[0]], landmarks[connection[1]]])
        
        points = np.array([(int(l.x * w), int(l.y * h)) for l in landmarks.landmark])
        hull = cv2.convexHull(points.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        return mask 

    def _analyze_color(self, region: np.ndarray) -> Dict:
        """Analyze color statistics in multiple color spaces"""
        lab_region = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)
        hsv_region = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        
        return {
            'rgb': {
                'mean': np.mean(region, axis=(0, 1)),
                'median': np.median(region, axis=(0, 1)),
                'percentiles': np.percentile(region, [5, 25, 50, 75, 95], axis=(0, 1))
            },
            'lab': {
                'mean': np.mean(lab_region, axis=(0, 1)),
                'median': np.median(lab_region, axis=(0, 1))
            },
            'hsv': {
                'mean': np.mean(hsv_region, axis=(0, 1)),
                'median': np.median(hsv_region, axis=(0, 1))
            }
        }

    def _process_single_image(self, image_path: str) -> Optional[Dict]:
        """Process a single image and return analysis results"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                logging.error(f"Failed to load image: {image_path}")
                return None

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_img)
            
            if not results.multi_face_landmarks:
                logging.warning(f"No faces detected in {image_path}")
                return None

            analysis = []
            for face_idx, landmarks in enumerate(results.multi_face_landmarks):
                mask = self._create_face_mask(rgb_img, landmarks)
                skin_region = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
                
                # Get non-zero pixels
                y, x = np.where(mask > 0)
                if len(y) == 0:
                    continue
                
                region_pixels = rgb_img[y, x]
                color_stats = self._analyze_color(region_pixels)
                
                analysis.append({
                    'face_id': face_idx,
                    'color_stats': color_stats,
                    'mask': mask,
                    'skin_region': skin_region
                })

            return {
                'image_path': image_path,
                'original_image': rgb_img,
                'analysis': analysis
            }

        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            return None

    def analyze_images(self, image_paths: List[str], output_dir: str = None) -> Dict:
        """Process multiple images with parallel execution"""
        results = {}
        
        with Pool(processes=self.num_processes) as pool:
            for result in pool.imap(self._process_single_image, image_paths):
                if result:
                    results[result['image_path']] = result
                    if output_dir:
                        self._generate_visualization(result, output_dir)
        
        return results

    def _generate_visualization(self, result: Dict, output_dir: str):
        """Generate visualization for analysis results"""
        plt.figure(figsize=(20, 10))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(result['original_image'])
        plt.title('Original Image')
        plt.axis('off')

        # Mask
        plt.subplot(1, 3, 2)
        plt.imshow(result['analysis'][0]['mask'], cmap='gray')
        plt.title('Face Mask')
        plt.axis('off')

        # Skin region
        plt.subplot(1, 3, 3)
        plt.imshow(result['analysis'][0]['skin_region'])
        plt.title('Skin Region')
        plt.axis('off')

        # Color swatches
        plt.figure(figsize=(10, 5))
        for idx, analysis in enumerate(result['analysis']):
            rgb_mean = analysis['color_stats']['rgb']['mean']
            lab_mean = analysis['color_stats']['lab']['mean']
            hsv_mean = analysis['color_stats']['hsv']['mean']
            
            # Create color swatches
            swatch_rgb = np.full((100, 100, 3), rgb_mean, dtype=np.uint8)
            swatch_lab = cv2.cvtColor(np.full((100, 100, 3), lab_mean, dtype=np.uint8), cv2.COLOR_LAB2RGB)
            swatch_hsv = cv2.cvtColor(np.full((100, 100, 3), hsv_mean, dtype=np.uint8), cv2.COLOR_HSV2RGB)
            
            plt.subplot(1, 3, 1)
            plt.imshow(swatch_rgb)
            plt.title(f'RGB Mean\n{np.round(rgb_mean)}')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(swatch_lab)
            plt.title(f'LAB Mean\n{np.round(lab_mean)}')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(swatch_hsv)
            plt.title(f'HSV Mean\n{np.round(hsv_mean)}')
            plt.axis('off')

        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/{Path(result['image_path']).stem}_{timestamp}.png"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Advanced Skin Tone Analysis')
    parser.add_argument('images', nargs='+', help='Input image paths')
    parser.add_argument('--output', '-o', help='Output directory for visualizations')
    parser.add_argument('--processes', '-p', type=int, 
                      default=cpu_count(),
                      help='Number of parallel processes')
    
    args = parser.parse_args()
    
    analyzer = SkinToneAnalyzer(num_processes=args.processes)
    results = analyzer.analyze_images(args.images, args.output)
    
    # Generate summary report
    summary = {
        'total_images': len(args.images),
        'processed_images': len(results),
        'average_colors': [],
        'color_variations': []
    }
    
    # Calculate aggregate statistics
    all_rgb = []
    for result in results.values():
        for analysis in result['analysis']:
            all_rgb.append(analysis['color_stats']['rgb']['mean'])
    
    if all_rgb:
        summary['average_colors'] = {
            'rgb': np.mean(all_rgb, axis=0).tolist(),
            'rgb_std': np.std(all_rgb, axis=0).tolist()
        }
    
    print("\nFinal Report:")
    print(f"Processed {summary['processed_images']}/{summary['total_images']} images")
    if summary['average_colors']:
        print("Average Skin Tone (RGB):", np.round(summary['average_colors']['rgb']))
        print("Standard Deviation:", np.round(summary['average_colors']['rgb_std']))

if __name__ == "__main__":
    main()