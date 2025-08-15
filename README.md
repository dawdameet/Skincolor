NeuraStyle

AI-Powered Skin Tone Analysis & Color Harmony

NeuraStyle is an advanced skin tone recognition and color recommendation system built with MediaPipe, OpenCV, and PyTorch. It detects accurate skin tones, eliminates noise (beards, goggles, shadows), and provides harmonious color suggestions based on color theory.

Features

Accurate Skin Tone Detection – Utilizes MediaPipe Face Mesh for precise face segmentation.

Noise Reduction – Excludes beard, goggles, and non-skin areas for higher accuracy.

Multi-Image Processing – Enhances precision by analyzing multiple images.

Color Recommendation – Suggests the best-matching colors based on detected skin tone.

Custom Correction Mode – Allows manual input if the detected color is incorrect.

Neural Network Training – Continuously improves accuracy using user feedback.

Installation
git clone https://github.com/dawdameet/NeuraStyle.git
cd NeuraStyle
pip install -r requirements.txt

Usage
1. Extract Skin Tone
python detect_skin.py --images path/to/image1.jpg path/to/image2.jpg


Use --exclude_beard to ignore the lower face area.

2. Color Harmony Recommendation
python suggest_colors.py --rgb 200 150 120


Input detected RGB values for personalized color recommendations.

3. Train Neural Network (Optional)
python train_correction.py --dataset path/to/dataset/


Improves model accuracy based on corrected inputs.

Example Output

Detected Skin Tone (RGB): [205, 160, 140]
Recommended Colors: #FF5733 (Warm Red), #FFD700 (Golden Yellow), #8A2BE2 (Deep Purple)

Future Enhancements

Real-time skin tone detection via webcam.

Advanced neural network for fine-tuned accuracy.

Personalized fashion and makeup recommendations.

Author

Developed by Meet Dawda
