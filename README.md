# **NeuraStyle** 🎨✨  
**AI-Powered Skin Tone Analysis & Color Harmony**  

NeuraStyle is an advanced **skin tone recognition and color recommendation** system using **MediaPipe, OpenCV, and PyTorch**. It detects accurate skin tones, eliminates noise (beards, goggles, shadows), and suggests **harmonious colors** using **color theory**.  

## **🚀 Features**  
✔ **Accurate Skin Tone Detection** – Uses MediaPipe Face Mesh for precise face segmentation.  
✔ **Noise Reduction** – Ignores beard, goggles, and non-skin areas for higher accuracy.  
✔ **Multi-Image Processing** – Increases precision by analyzing multiple images.  
✔ **Color Recommendation** – Suggests best-matching colors based on skin tone.  
✔ **Custom Correction Mode** – Allows manual input if detected color is incorrect.  
✔ **Neural Network Training** – Improves accuracy with user feedback.  

## **🛠 Installation**  
```bash
git clone https://github.com/dawdameet/NeuraStyle.git
cd NeuraStyle
pip install -r requirements.txt
```

## **▶ Usage**  
### **1️⃣ Extract Skin Tone**  
```bash
python detect_skin.py --images path/to/image1.jpg path/to/image2.jpg
```
🔹 Add `--exclude_beard` to ignore lower face area.  

### **2️⃣ Color Harmony Recommendation**  
```bash
python suggest_colors.py --rgb 200 150 120
```
🔹 Input detected RGB values for personalized color recommendations.  

### **3️⃣ Train Neural Network (Optional)**  
```bash
python train_correction.py --dataset path/to/dataset/
```
🔹 Improves model accuracy based on corrected inputs.  

## **🎨 Example Output**  
**Detected Skin Tone (RGB):** `[205, 160, 140]`  
**Recommended Colors:** `#FF5733 (Warm Red)`, `#FFD700 (Golden Yellow)`, `#8A2BE2 (Deep Purple)`

## **🧠 Future Enhancements**  
✅ Real-time skin tone detection via webcam.  
✅ Advanced neural network for fine-tuned accuracy.  
✅ Personalized fashion & makeup recommendations.  

## **Author**  
Developed by **[Meet Dawda](https://github.com/dawdameet)** 🚀🔥  
