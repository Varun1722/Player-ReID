# 🏈 Football Analysis System (Player Reidentification)

This project performs football match analysis using object detection and multi-object tracking (MOT). Leveraging YOLO-based detection along with DeepSORT and BoT-SORT tracking, it enables detailed analysis of player movement and behavior in football match videos.
It focuses specifically on Player Reidentification in soccer reels and videos tackling Occlusion and the case when a player re-emerges after disappearing for a specific amount of frames. 
---

## ✅ Features

* ⚽ Player detection from broadcast or tactical videos
* 🧠 DeepSORT and BoT-SORT tracking support
* ✏️ Annotated detection and tracking output
* 🎥 Supports `.mp4` and `.avi` video formats
* 📃 Saves tracking data and output videos

---

## 📊 Project Structure (simplified)

```
.
├── annotation/
├── football_analysis/
│   ├── tracker/
│   ├── utils/
│   └── output_videos/
├── model/
├── tracker_BoTSort/
├── tracker_deepsort/
├── videos/
├── main.py
├── detection.py
├── requirements.txt
└── README.md
```
NOTE: The main folder is football_analysis/. Others was me just trying different options.
---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/football-analysis.git
cd football-analysis
```

### 2. Set up a virtual environment (recommended)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download and place model weights

Create a `model/` directory if not present and add:

* `best.pt` (YOLO model for detection) 
Note: This is only for the company who provided me with this assignment. Others just download `yolov11m.pt` or any other YOLO model. 
* `osnet_ain_ms_d_c.pth.tar` (OSNet for DeepSORT)

You can download OSNet weights from [Open-ReID Model Zoo](https://github.com/KaiyangZhou/deep-person-reid#model-zoo).

---

## 📹 Input Videos

Place your input videos in the `videos/` folder.

Supported formats: `.mp4`, `.avi`

---

## 🚀 Usage

### 1. Full pipeline (detection + tracking)

```bash
cd football-analysis
python main.py
```

Output video and results are saved in:

```
football_analysis/output_videos/
```

## 🔧 Recommended .gitignore

```
# Python cache
__pycache__/
*.pyc

# Model weights
model/*.pt
model/*.pth.tar

# Binary output  (Recommended if output videos are large)
*.avi
*.mp4
*.pkl
football_analysis/output_videos/
football_analysis/runs/
```

---

## 🤝 Contribution

Pull requests are welcome! For significant changes, please open an issue first to discuss.

---

## 💌 Contact

For questions, feedback, or collaboration: [varun170402@gmail.com](mailto:varun170402@gmail.com)
